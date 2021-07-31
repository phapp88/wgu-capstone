import streamlit as st

from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server

class SessionState:

  def __init__(self, session, hash_funcs):
    self.__dict__['state'] = {
      'data': {},
      'hash': None,
      'hasher': _CodeHasher(hash_funcs),
      'is_rerun': False,
      'session': session
    }

  def __call__(self, **kwargs):
    for item, value in kwargs.items():
      if item not in self.state['data']:
        self.state['data'][item] = value
  
  def __getitem__(self, item):
    return self.state['data'].get(item, None)
  
  def __getattr__(self, item):
    return self.state['data'].get(item, None)
  
  def __setitem__(self, item, value):
    self.state['data'][item] = value
  
  def __setattr__(self, item, value):
    self.state['data'][item] = value
  
  def clear(self):
    self.state['data'].clear()
    self.state['session'].request_rerun()

  def sync(self):
    if self.state['is_rerun']:
      self.state['is_rerun'] = False
    
    elif self.state['hash'] is not None:
      if self.state['hash'] != self.state['hasher'].to_bytes(self.state['data'], None):
        self.state['is_rerun'] = True
        self.state['session'].request_rerun()
    
    self.state['hash'] = self.state['hasher'].to_bytes(self.state['data'], None)

def get_session():
  session_id = get_report_ctx().session_id
  session_info = Server.get_current()._get_session_info(session_id)

  if session_info is None:
    raise RuntimeError("Couldn't get your Streamlit Session object.")

  return session_info.session

def get_state(hash_funcs = None):
  session = get_session()

  if not hasattr(session, 'custom_session_state'):
    session.custom_session_state = SessionState(session, hash_funcs)
  
  return session.custom_session_state
  