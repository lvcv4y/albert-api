from typing import List, Optional

from app.schemas.models import Model as ModelSchema
from app.utils.exceptions import ModelNotFoundException

from app.helpers.models.routers import ModelRouter


class ModelRegistry:
    def __init__(self, routers: List[ModelRouter]) -> None:
        self.models = list()
        self.aliases = dict()

        for model in routers:
            if "name" not in model.__dict__:  # no clients available
                continue

            self.__dict__[model.name] = model
            self.models.append(model.name)

            for alias in model.aliases:
                self.aliases[alias] = model.name

    def __call__(self, model: str) -> ModelRouter:
        model = self.aliases.get(model, model)

        if model in self.models:
            return self.__dict__[model]
        raise ModelNotFoundException()

    def list(self, model: Optional[str] = None) -> List[ModelSchema]:
        data = list()
        models = [model] if model else self.models
        for model in models:
            model = self.__call__(model=model)

            data.append(
                ModelSchema(
                    id=model.name,
                    type=model.type,
                    max_context_length=model.max_context_length,
                    owned_by=model.owned_by,
                    created=model.created,
                    aliases=model.aliases,
                    costs={"prompt_tokens": model.cost_prompt_tokens, "completion_tokens": model.cost_completion_tokens},
                )
            )

        return data
