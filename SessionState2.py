import streamlit as st
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server

class SessionState(object):
  def __init__(self, session, **kwargs):
    self.session = session

    for key, val in kwargs.items():
      setattr(self, key, val)
  
@st.cache(allow_output_mutation = True)
def get_session(id, session, **kwargs):
  return SessionState(session, **kwargs)

def get(**kwargs):
  session_id = get_report_ctx().session_id
  session = Server.get_current()._get_session_info(session_id).session

  return get_session(session_id, session, **kwargs)
