"""Start the local demo, optionally using project-local UI dependencies."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
local_deps = root / '.demo-deps'
if local_deps.is_dir():
    sys.path.insert(0,str(local_deps))
try:
    from streamlit.web import cli
except ImportError as exc:
    raise SystemExit('Thiếu Streamlit. Cài bằng: python -m pip install streamlit==1.67.0') from exc

if __name__ == '__main__':
    sys.argv = ['streamlit','run',str(root/'app.py'),'--server.address=127.0.0.1',
                '--server.port=8501','--server.headless=true','--global.developmentMode=false',*sys.argv[1:]]
    cli.main()
