from kedro.framework.hooks import hook_impl
import sklearn


class ProjectHooks:
    @hook_impl
    def after_context_created(self, context) -> None:
        sklearn.set_config(transform_output="pandas")
