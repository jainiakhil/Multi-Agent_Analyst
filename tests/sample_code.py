"""Sample module used for validating Python AST inspection."""


class DataPipeline:
    """Processes incoming data streams."""

    def __init__(self, source: str) -> None:
        self.source = source

    def process(self, batch_size: int = 100) -> list:
        return [self.source] * batch_size


def compute_metrics(accuracy: float, loss: float) -> dict:
    """Calculates performance summary metrics."""
    return {"acc": accuracy, "loss": loss}
