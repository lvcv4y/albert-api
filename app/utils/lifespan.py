from contextlib import asynccontextmanager
import traceback

from coredis import ConnectionPool, Redis
from fastapi import FastAPI
from sqlalchemy import select, insert, delete  # Integer, cast, delete, distinct, func, insert, or_, select, update

from app.clients.mcp import SecretShellMCPBridgeClient
from app.clients.model import BaseModelClient as ModelClient
from app.clients.parser import BaseParserClient as ParserClient
from app.clients.vector_store import BaseVectorStoreClient as VectorStoreClient
from app.clients.web_search import BaseWebSearchClient as WebSearchClient
from app.helpers._documentmanager import DocumentManager
from app.helpers._identityaccessmanager import IdentityAccessManager
from app.helpers._limiter import Limiter
from app.helpers._parsermanager import ParserManager
from app.helpers._usagetokenizer import UsageTokenizer
from app.helpers._websearchmanager import WebSearchManager
from app.helpers.agents import AgentsManager
from app.helpers.models import ModelRegistry
from app.helpers.models.routers import ModelRouter
from app.utils.context import global_context
from app.utils.logging import init_logger
from app.utils.settings import settings

from app.sql.session import get_db_session
from app.sql.models import ModelRouter as ModelRouterTable
from app.sql.models import ModelRouterAlias as ModelRouterAliasTable
from app.sql.models import ModelClient as ModelClientTable


logger = init_logger(name=__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to initialize clients (models API and databases)."""

    # setup redis
    assert settings.databases.redis is not None, "Redis database connection parameters must be set in configuration."
    redis = ConnectionPool(**settings.databases.redis.args)
    redis_test_client = Redis(connection_pool=redis)
    assert (await redis_test_client.ping()).decode("ascii") == "PONG", "Redis database is not reachable."

    assert settings.databases.sql is not None, "SQL database connection parameters must be set in configuration."

    routers = []
    async for session in get_db_session():
        # Get all ModelRouter rows and cnvert it from a list of 1-dimensional vectors to a list of ModelRouters
        db_routers = [row[0] for row in (await session.execute(select(ModelRouterTable))).fetchall()]

        if not db_routers:
            logger.warning(msg="no modelrouter found in database. initializing from configuration file.")
            break

        for router in db_routers:
            # Get all ModelAlias rows and convert from a list of 1-dimensional vectors to a list of values
            db_aliases = [
                row[0]
                for row in (await session.execute(select(ModelRouterAliasTable).where(ModelRouterAliasTable.model_router_id == router.id))).fetchall()
            ]

            if not db_aliases:
                logger.info(msg=f"no alias found in database for modelrouter {router.id}.")

            db_clients = [
                row[0] for row in (await session.execute(select(ModelClientTable).where(ModelClientTable.model_router_id == router.id))).fetchall()
            ]

            if not db_clients:
                logger.fatal(msg=f"no client model found in database for modelrouter {router['id']}.")
                # @TODO : verify that it breaks here

            clients = []
            for client in db_clients:
                # clients.append(ModelClient(model=client.model, costs=client.costs, carbon=client.carbon))
                pass
                # @TODO functional client creation from database

            routers.append(
                ModelRouter(id=router["id"], type=router["type"], aliases=db_aliases, routing_strategy=router["routing_strategy"], clients=clients)
            )

    # Global context: models
    routers = []
    for model in settings.models:
        clients = []
        for client in model.clients:
            try:
                # model client can be not reatachable to API start up
                client = (
                    await ModelClient.import_module(type=client.type, connection_pool=redis, model_name=client.model, api_url=client.args.api_url)
                )(
                    model=client.model,
                    costs=client.costs,
                    carbon=client.carbon,
                    connection_pool=redis,
                    **client.args.model_dump(),
                )
                clients.append(client)
            except Exception:
                logger.debug(msg=traceback.format_exc())
                continue
        if not clients:
            logger.error(msg=f"skip model {model.id} (0/{len(model.clients)} clients).")
            if settings.web_search and model.id == settings.web_search.query_model:
                raise ValueError(f"Web search model ({model.id}) must be reachable.")
            if settings.databases.vector_store and model.id == settings.databases.vector_store.model:
                raise ValueError(f"Vector store embedding model ({model.id}) must be reachable.")
            continue

        logger.info(msg=f"add model {model.id} ({len(clients)}/{len(model.clients)} clients).")
        model = model.model_dump()
        model["clients"] = clients
        routers.append(ModelRouter(**model))

    async for session in get_db_session():
        for router in routers:
            result = await session.execute(delete(ModelRouterTable).where(ModelRouterTable.id == router.id))  # .fetchone()
            result = await session.execute(delete(ModelRouterAliasTable).where(ModelRouterAliasTable.model_router_id == router.id))  # .fetchone()
            await session.commit()

            result = (await session.execute(select(ModelRouterTable).where(ModelRouterTable.id == router.id))).fetchone()

            if not result:
                # @TODO Check why routing strategy is private
                await session.execute(
                    insert(ModelRouterTable).values(id=router.id, type=router.type, routing_strategy=router._routing_strategy, from_config=True)
                )

                for alias in router.aliases:
                    await session.execute(insert(ModelRouterAliasTable).values(alias=alias, model_router_id=router.id))

                # @TODO Check why clients is private
                for client in router._clients:
                    await session.execute(
                        insert(ModelClientTable).values(
                            model=client.model,
                            model_router_id=router.id,
                            type=type(client).__name__.removesuffix("ModelClient").lower(),
                            prompt_token_cost=client.costs.prompt_tokens,
                            completion_token_cost=client.costs.completion_tokens,
                            total_parameters=client.carbon.total_params,
                            active_parameters=client.carbon.active_params,
                            model_zone=client.carbon.model_zone,
                            api_url=client.api_url,
                            api_key=client.api_key,
                            timeout=client.timeout,
                        )
                    )
                await session.commit()

    global_context.models = ModelRegistry(routers=routers)

    # Global context: iam
    global_context.iam = IdentityAccessManager()

    # Global context: limiter
    global_context.limiter = Limiter(connection_pool=redis, strategy=settings.auth.limiting_strategy)

    # Global context: tokenizer
    global_context.tokenizer = UsageTokenizer(tokenizer=settings.general.tokenizer)

    # Global context: mcp
    mcp_bridge = SecretShellMCPBridgeClient(mcp_bridge_url=settings.mcp.mcp_bridge_url)
    global_context.mcp.agents_manager = AgentsManager(mcp_bridge=mcp_bridge, model_registry=global_context.models)

    # Global context: documents

    ## documents dependency: web search
    web_search = WebSearchClient.import_module(
        websearch_type=settings.web_search.client.type)(**settings.web_search.client.args.model_dump()) if settings.web_search else None  # fmt: off
    if web_search:
        web_search = WebSearchManager(
            web_search=web_search,
            model=global_context.models(model=settings.web_search.query_model),
            limited_domains=settings.web_search.limited_domains,
            user_agent=settings.web_search.user_agent,
        )

    ## documents dependency: parser
    parser = ParserClient.import_module(parser_type=settings.parser.type)(**settings.parser.args.model_dump()) if settings.parser else None
    parser = ParserManager(parser=parser)

    ## documents dependency: vector store
    vector_store = None
    if settings.databases.vector_store:
        vector_store = VectorStoreClient.import_module(
            database_type=settings.databases.vector_store.type
        )(
            **settings.databases.vector_store.args,
            model=global_context.models(model=settings.databases.vector_store.model)
        )  # fmt: off

    if vector_store:
        assert await vector_store.check(), "Vector store database is not reachable."

    ## documents dependency: multi agents
    multi_agents_model = global_context.models(model=settings.multi_agents_search.model) if settings.multi_agents_search else None
    multi_agents_reranker_model=global_context.models(model=settings.multi_agents_search.ranker_model) if settings.multi_agents_search else None  # fmt: off

    global_context.documents = DocumentManager(
        vector_store=vector_store,
        parser=parser,
        web_search=web_search,
        multi_agents_model=multi_agents_model,
        multi_agents_reranker_model=multi_agents_reranker_model,
    )

    yield

    # cleanup resources when app shuts down
    if vector_store:
        await vector_store.close()
