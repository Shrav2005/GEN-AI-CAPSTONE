"""Small SQLite-backed conversation store for durable browser sessions."""
import json
import sqlite3
import uuid
from pathlib import Path
from contextlib import contextmanager
from .config import settings

CONTEXT_MESSAGE_LIMIT=24

class ConversationStore:
    def __init__(self, path=None):
        self.path=Path(path or settings.data_path)/"conversations.sqlite3"
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS conversations (session_id TEXT PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
            db.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, sources TEXT NOT NULL DEFAULT '[]', fallback INTEGER NOT NULL DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
            db.execute("CREATE INDEX IF NOT EXISTS messages_session_id ON messages(session_id,id)")
    @contextmanager
    def _db(self):
        db=sqlite3.connect(self.path,timeout=15)
        db.row_factory=sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
    def create(self):
        key=str(uuid.uuid4())
        with self._db() as db: db.execute("INSERT INTO conversations(session_id) VALUES(?)",(key,))
        return key
    def exists(self,key):
        with self._db() as db: return db.execute("SELECT 1 FROM conversations WHERE session_id=?",(key,)).fetchone() is not None
    def history(self,key,limit=CONTEXT_MESSAGE_LIMIT):
        with self._db() as db: rows=db.execute("SELECT role,content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",(key,limit)).fetchall()
        return [dict(row) for row in reversed(rows)]
    def messages(self,key,limit=100):
        with self._db() as db: rows=db.execute("SELECT role,content,sources,fallback FROM (SELECT * FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?) ORDER BY id",(key,limit)).fetchall()
        return [{"role":r["role"],"text":r["content"],"sources":json.loads(r["sources"]),"fallback":bool(r["fallback"])} for r in rows]
    def append_turn(self,key,user,result):
        if not self.exists(key):
            with self._db() as db: db.execute("INSERT OR IGNORE INTO conversations(session_id) VALUES(?)",(key,))
        with self._db() as db:
            db.executemany("INSERT INTO messages(session_id,role,content,sources,fallback) VALUES(?,?,?,?,?)",[(key,"user",user,"[]",0),(key,"assistant",result.get("answer",""),json.dumps(result.get("sources",[])),int(bool(result.get("fallback"))))])
            db.execute("DELETE FROM messages WHERE session_id=? AND id NOT IN (SELECT id FROM messages WHERE session_id=? ORDER BY id DESC LIMIT 100)",(key,key))

conversations=ConversationStore()
