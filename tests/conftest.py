import os
import tempfile

# 테스트는 실제 홈이 아닌 임시 .freecloud 를 쓰도록 격리(쿨다운/시크릿 파일 오염 방지).
os.environ.setdefault("FREECLOUD_HOME", tempfile.mkdtemp(prefix="fc-test-home-"))
os.environ.setdefault("PYTHONUTF8", "1")
