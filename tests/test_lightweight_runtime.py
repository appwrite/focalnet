import subprocess
import sys


def test_human_runtime_does_not_import_training_stack():
    command = (
        "import sys; "
        "from focalnet.human_runtime import HumanCropPredictor; "
        "assert 'torch' not in sys.modules; "
        "assert 'timm' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
