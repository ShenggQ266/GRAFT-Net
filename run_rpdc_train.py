from types import SimpleNamespace

from Semi_GPR.config import CONFIG
from Semi_GPR.train import main


if __name__ == "__main__":
    main(SimpleNamespace(**CONFIG))
