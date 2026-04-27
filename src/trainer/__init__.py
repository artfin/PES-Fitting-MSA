from .base import TrainingBase
from .optimization import TrainingOptimizationMixin
from .gradients import TrainingGradientsMixin
from .diagnostics import TrainingDiagnosticsMixin
from .loop import TrainingLoopMixin
from .batching import TrainingBatchingMixin

class Training(TrainingBase, TrainingOptimizationMixin, TrainingGradientsMixin,
               TrainingDiagnosticsMixin, TrainingLoopMixin, TrainingBatchingMixin):
    pass
