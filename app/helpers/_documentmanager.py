from functools import wraps
from itertools import batched
import logging
import time
from typing import Callable, List, Optional
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from langchain_text_splitters import Language
from sqlalchemy import Integer, cast, delete, distinct, func, insert, or_, select, update
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.vector_store import BaseVectorStoreClient
from app.helpers.data.chunkers import NoSplitter, RecursiveCharacterTextSplitter
from app.helpers.models.routers import ModelRouter
from app.schemas.chunks import Chunk
from app.schemas.collections import Collection, CollectionVisibility
from app.schemas.documents import Chunker, Document
from app.schemas.parse import ParsedDocument, ParsedDocumentOutputFormat
from app.schemas.search import Search, SearchMethod
from app.sql.models import Collection as CollectionTable
from app.sql.models import Document as DocumentTable
from app.sql.models import User as UserTable
from app.utils.exceptions import (
    ChunkingFailedException,
    CollectionNotFoundException,
    DocumentNotFoundException,
    MultiAgentSearchNotAvailableException,
    VectorizationFailedException,
)
from app.utils.variables import ENDPOINT__EMBEDDINGS

from ._multiagentmanager import MultiAgentManager
from ._parsermanager import ParserManager
from ._websearchmanager import WebSearchManager

logger = logging.getLogger(__name__)


def check_dependencies(*, dependencies: List[str]) -> Callable:
    """
    Decorator to return a 400 error to the user if the endpoint calls a feature that requires an uninitialized dependency.
    """

    def decorator(method: Callable) -> Callable:
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if "vector_store" in dependencies and not self.vector_store:
                raise HTTPException(status_code=400, detail="Feature not available: vector store is not initialized.")
            if "web_search_manager" in dependencies and not self.web_search_manager:
                raise HTTPException(status_code=400, detail="Feature not available: web search is not initialized.")
            if "parser_manager" in dependencies and not self.parser_manager:
                raise HTTPException(status_code=400, detail="Feature not available: parser is not initialized.")
            if "multi_agent_manager" in dependencies and not self.multi_agent_manager:
                raise HTTPException(status_code=400, detail="Feature not available: multi agents is not initialized.")

            return method(self, *args, **kwargs)

        return wrapper

    return decorator


class DocumentManager:
    BATCH_SIZE = 32

    def __init__(
        self,
        vector_store: BaseVectorStoreClient,
        vector_store_model: ModelRouter,
        parser_manager: ParserManager,
        web_search_manager: Optional[WebSearchManager] = None,
        multi_agent_manager: Optional[MultiAgentManager] = None,
    ) -> None:
        self.vector_store = vector_store
        self.vector_store_model = vector_store_model
        self.web_search_manager = web_search_manager
        self.parser_manager = parser_manager
        self.multi_agent_manager = multi_agent_manager

    @check_dependencies(dependencies=["vector_store"])
    async def create_collection(self, session: AsyncSession, user_id: int, name: str, visibility: CollectionVisibility, description: Optional[str] = None) -> int:  # fmt: off
        result = await session.execute(
            statement=insert(table=CollectionTable)
            .values(name=name, user_id=user_id, visibility=visibility, description=description)
            .returning(CollectionTable.id)
        )
        collection_id = result.scalar_one()
        await session.commit()

        await self.vector_store.create_collection(collection_id=collection_id, vector_size=self.vector_store_model._vector_size)

        return collection_id

    @check_dependencies(dependencies=["vector_store"])
    async def delete_collection(self, session: AsyncSession, user_id: int, collection_id: int) -> None:
        # check if collection exists
        result = await session.execute(
            statement=select(CollectionTable.id).where(CollectionTable.id == collection_id).where(CollectionTable.user_id == user_id)
        )
        try:
            result.scalar_one()
        except NoResultFound:
            raise CollectionNotFoundException()

        # delete the collection
        await session.execute(statement=delete(table=CollectionTable).where(CollectionTable.id == collection_id))
        await session.commit()

        # delete the collection from vector store
        await self.vector_store.delete_collection(collection_id=collection_id)

    @check_dependencies(dependencies=["vector_store"])
    async def update_collection(self, session: AsyncSession, user_id: int, collection_id: int, name: Optional[str] = None, visibility: Optional[CollectionVisibility] = None, description: Optional[str] = None) -> None:  # fmt: off
        # check if collection exists
        result = await session.execute(
            statement=select(CollectionTable)
            .join(target=UserTable, onclause=UserTable.id == CollectionTable.user_id)
            .where(CollectionTable.id == collection_id)
            .where(UserTable.id == user_id)
        )
        try:
            collection = result.scalar_one()
        except NoResultFound:
            raise CollectionNotFoundException()

        name = name if name is not None else collection.name
        visibility = visibility if visibility is not None else collection.visibility
        description = description if description is not None else collection.description

        await session.execute(
            statement=update(table=CollectionTable)
            .values(name=name, visibility=visibility, description=description)
            .where(CollectionTable.id == collection.id)
        )
        await session.commit()

    @check_dependencies(dependencies=["vector_store"])
    async def get_collections(self, session: AsyncSession, user_id: int, collection_id: Optional[int] = None, include_public: bool = True, offset: int = 0, limit: int = 10) -> List[Collection]:  # fmt: off
        # Query basic collection data
        statement = (
            select(
                CollectionTable.id,
                CollectionTable.name,
                UserTable.name.label("owner"),
                CollectionTable.visibility,
                CollectionTable.description,
                func.count(distinct(DocumentTable.id)).label("documents"),
                cast(func.extract("epoch", CollectionTable.created_at), Integer).label("created_at"),
                cast(func.extract("epoch", CollectionTable.updated_at), Integer).label("updated_at"),
            )
            .outerjoin(DocumentTable, CollectionTable.id == DocumentTable.collection_id)
            .outerjoin(UserTable, CollectionTable.user_id == UserTable.id)
            .group_by(CollectionTable.id, UserTable.name)
            .offset(offset=offset)
            .limit(limit=limit)
        )

        if collection_id:
            statement = statement.where(CollectionTable.id == collection_id)
        if include_public:
            statement = statement.where(or_(CollectionTable.user_id == user_id, CollectionTable.visibility == CollectionVisibility.PUBLIC))
        else:
            statement = statement.where(CollectionTable.user_id == user_id)

        result = await session.execute(statement=statement)
        collections = [Collection(**row._asdict()) for row in result.all()]

        if collection_id and len(collections) == 0:
            raise CollectionNotFoundException()

        return collections

    @check_dependencies(dependencies=["vector_store"])
    async def create_document(
        self,
        session: AsyncSession,
        user_id: int,
        collection_id: int,
        document: ParsedDocument,
        chunker: Chunker,
        chunk_size: int,
        chunk_overlap: int,
        length_function: Callable,
        chunk_min_size: int,
        is_separator_regex: Optional[bool] = None,
        separators: Optional[List[str]] = None,
        preset_separators: Optional[Language] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        # check if collection exists and prepare document chunks in a single transaction
        result = await session.execute(
            statement=select(CollectionTable).where(CollectionTable.id == collection_id).where(CollectionTable.user_id == user_id)
        )
        try:
            result.scalar_one()
        except NoResultFound:
            raise CollectionNotFoundException()

        try:
            chunks = self._split(
                document=document,
                chunker=chunker,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=length_function,
                is_separator_regex=is_separator_regex,
                separators=separators,
                chunk_min_size=chunk_min_size,
                preset_separators=preset_separators,
                metadata=metadata,
            )
        except Exception as e:
            logger.exception(msg=f"Error during document splitting: {e}")
            raise ChunkingFailedException(detail=f"Chunking failed: {e}")

        document_name = document.data[0].metadata.document_name
        try:
            result = await session.execute(
                statement=insert(table=DocumentTable).values(name=document_name, collection_id=collection_id).returning(DocumentTable.id)
            )
        except Exception as e:
            if "foreign key constraint" in str(e).lower() or "fkey" in str(e).lower():
                raise CollectionNotFoundException(detail=f"Collection {collection_id} no longer exists")
        document_id = result.scalar_one()
        await session.commit()

        for chunk in chunks:
            chunk.metadata["collection_id"] = collection_id
            chunk.metadata["document_id"] = document_id
            chunk.metadata["document_created_at"] = round(time.time())
        try:
            await self._upsert(chunks=chunks, collection_id=collection_id)
        except Exception as e:
            logger.exception(msg=f"Error during document creation: {e}")
            await self.delete_document(session=session, user_id=user_id, document_id=document_id)
            raise VectorizationFailedException(detail=f"Vectorization failed: {e}")

        return document_id

    @check_dependencies(dependencies=["vector_store"])
    async def get_documents(self, session: AsyncSession, user_id: int, collection_id: Optional[int] = None, document_id: Optional[int] = None, offset: int = 0, limit: int = 10) -> List[Document]:  # fmt: off
        statement = (
            select(
                DocumentTable.id,
                DocumentTable.name,
                DocumentTable.collection_id,
                cast(func.extract("epoch", DocumentTable.created_at), Integer).label("created_at"),
            )
            .offset(offset=offset)
            .limit(limit=limit)
            .outerjoin(CollectionTable, DocumentTable.collection_id == CollectionTable.id)
            .where(or_(CollectionTable.user_id == user_id, CollectionTable.visibility == CollectionVisibility.PUBLIC))
        )
        if collection_id:
            statement = statement.where(DocumentTable.collection_id == collection_id)
        if document_id:
            statement = statement.where(DocumentTable.id == document_id)

        result = await session.execute(statement=statement)
        documents = [Document(**row._asdict()) for row in result.all()]

        if document_id and len(documents) == 0:
            raise DocumentNotFoundException()

        # chunks count
        for document in documents:
            document.chunks = await self.vector_store.get_chunk_count(collection_id=document.collection_id, document_id=document.id)

        return documents

    @check_dependencies(dependencies=["vector_store"])
    async def delete_document(self, session: AsyncSession, user_id: int, document_id: int) -> None:
        # check if document exists
        result = await session.execute(
            statement=select(DocumentTable)
            .join(CollectionTable, DocumentTable.collection_id == CollectionTable.id)
            .where(DocumentTable.id == document_id)
            .where(CollectionTable.user_id == user_id)
        )
        try:
            document = result.scalar_one()
        except NoResultFound:
            raise DocumentNotFoundException()

        await session.execute(statement=delete(table=DocumentTable).where(DocumentTable.id == document_id))
        await session.commit()

        # delete the document from vector store
        await self.vector_store.delete_document(collection_id=document.collection_id, document_id=document_id)

    @check_dependencies(dependencies=["vector_store"])
    async def get_chunks(
        self,
        session: AsyncSession,
        user_id: int,
        document_id: int,
        chunk_id: Optional[int] = None,
        offset: int = 0,
        limit: int = 10,
    ) -> List[Chunk]:
        # check if document exists
        result = await session.execute(
            statement=select(DocumentTable)
            .join(CollectionTable, DocumentTable.collection_id == CollectionTable.id)
            .where(DocumentTable.id == document_id)
            .where(CollectionTable.user_id == user_id)
        )
        try:
            document = result.scalar_one()
        except NoResultFound:
            raise DocumentNotFoundException()

        chunks = await self.vector_store.get_chunks(
            collection_id=document.collection_id,
            document_id=document_id,
            offset=offset,
            limit=limit,
            chunk_id=chunk_id,
        )

        return chunks

    @check_dependencies(dependencies=["parser_manager"])
    async def parse_file(
        self,
        file: UploadFile,
        output_format: Optional[ParsedDocumentOutputFormat] = None,
        force_ocr: Optional[bool] = None,
        page_range: str = "",
        paginate_output: Optional[bool] = None,
        use_llm: Optional[bool] = None,
    ) -> ParsedDocument:
        return await self.parser_manager.parse_file(
            file=file, output_format=output_format, force_ocr=force_ocr, page_range=page_range, paginate_output=paginate_output, use_llm=use_llm
        )

    @check_dependencies(dependencies=["vector_store"])
    async def search_chunks(
        self,
        session: AsyncSession,
        collection_ids: List[int],
        user_id: int,
        prompt: str,
        method: str,
        k: int,
        rff_k: int,
        score_threshold: float = 0.0,
        web_search: bool = False,
        web_search_k: int = 5,
    ) -> List[Search]:
        web_collection_id = None
        if web_search:
            web_collection_id = await self._create_web_collection(session=session, user_id=user_id, prompt=prompt, k=web_search_k)
        if web_collection_id:
            collection_ids.append(web_collection_id)

        # check if collections exist
        for collection_id in collection_ids:
            result = await session.execute(
                statement=select(CollectionTable)
                .where(CollectionTable.id == collection_id)
                .where(or_(CollectionTable.user_id == user_id, CollectionTable.visibility == CollectionVisibility.PUBLIC))
            )
            try:
                result.scalar_one()
            except NoResultFound:
                raise CollectionNotFoundException(detail=f"Collection {collection_id} not found.")

        if not collection_ids:
            return []  # to avoid a request to create a query vector

        response = await self._create_embeddings(input=[prompt])
        query_vector = response[0]

        _method = method
        if method == SearchMethod.MULTIAGENT:
            _method = self.vector_store.default_method
            k = k * 4

        searches = await self.vector_store.search(
            method=_method,
            collection_ids=collection_ids,
            query_prompt=prompt,
            query_vector=query_vector,
            k=k,
            rff_k=rff_k,
            score_threshold=score_threshold,
        )
        if method == SearchMethod.MULTIAGENT:
            if not self.multi_agent_manager:
                raise MultiAgentSearchNotAvailableException()
            searches = await self.multi_agent_manager.search(searches=searches, prompt=prompt)

        if web_collection_id:
            await self.delete_collection(session=session, user_id=user_id, collection_id=web_collection_id)

        return searches

    @check_dependencies(dependencies=["web_search_manager"])
    async def _create_web_collection(
        self,
        session: AsyncSession,
        user_id: int,
        prompt: str,
        k: int = 5,
    ) -> Optional[int]:
        web_query = await self.web_search_manager.get_web_query(prompt=prompt)
        web_results = await self.web_search_manager.get_results(query=web_query, k=k)
        collection_id = None
        if web_results:
            collection_id = await self.create_collection(
                session=session,
                name=f"tmp_web_collection_{uuid4()}",
                user_id=user_id,
                visibility=CollectionVisibility.PRIVATE,
            )
            for file in web_results:
                document = await self.parse_file(
                    file=file,
                    output_format=ParsedDocumentOutputFormat.MARKDOWN.value,
                    force_ocr=False,
                    page_range="",
                    paginate_output=False,
                    use_llm=False,
                )
                await self.create_document(
                    session=session,
                    user_id=user_id,
                    collection_id=collection_id,
                    document=document,
                    chunker=Chunker.RECURSIVE_CHARACTER_TEXT_SPLITTER,
                    chunk_overlap=0,
                    chunk_min_size=20,
                    chunk_size=4000,
                    length_function=len,
                    preset_separators=Language.MARKDOWN.value,
                )

        return collection_id

    def _split(
        self,
        document: ParsedDocument,
        chunker: Chunker,
        chunk_size: int,
        chunk_min_size: int,
        chunk_overlap: int,
        length_function: Callable,
        separators: Optional[List[str]] = None,
        is_separator_regex: Optional[bool] = None,
        preset_separators: Optional[Language] = None,
        metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        if chunker == Chunker.RECURSIVE_CHARACTER_TEXT_SPLITTER:
            chunker = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_min_size=chunk_min_size,
                chunk_overlap=chunk_overlap,
                length_function=length_function,
                separators=separators,
                is_separator_regex=is_separator_regex,
                preset_separators=preset_separators,
                metadata=metadata,
            )
        else:  # Chunker.NoSplitter
            chunker = NoSplitter(chunk_min_size=chunk_min_size, preset_separators=preset_separators, metadata=metadata)

        chunks = chunker.split_document(document=document)

        return chunks

    async def _create_embeddings(self, input: List[str]) -> list[float] | list[list[float]] | dict:
        client = self.vector_store_model.get_client(endpoint=ENDPOINT__EMBEDDINGS)
        response = await client.forward_request(
            method="POST",
            json={"input": input, "model": self.vector_store_model.name, "encoding_format": "float"},
        )

        return [vector["embedding"] for vector in response.json()["data"]]

    async def _upsert(self, chunks: List[Chunk], collection_id: int) -> None:
        batches = batched(iterable=chunks, n=self.BATCH_SIZE)
        for batch in batches:
            # create embeddings
            texts = [chunk.content for chunk in batch]
            embeddings = await self._create_embeddings(input=texts)

            # insert chunks and vectors
            await self.vector_store.upsert(collection_id=collection_id, chunks=batch, embeddings=embeddings)
