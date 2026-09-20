from yoyo import step

from .helpers.config import migrate


__depends__ = set()
__transactional__ = False
steps = [step(migrate)]
