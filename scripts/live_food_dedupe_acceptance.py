"""Targeted live follow-up after full-trip acceptance; no unnecessary research tools."""
from pathlib import Path
from dataclasses import replace
import json
from langchain_core.messages import HumanMessage
from travel_agent.config import Settings
from travel_agent.graph.builder import build_graph
from travel_agent.persistence import Database
from travel_agent.cli import _trace_redact

root=Path('output/revision-acceptance');db=Database(root/'danang4.sqlite3')
settings=replace(Settings.from_env(),database_path=db.path,output_dir=root,tool_mode='real',require_real_model=True)
graph,tools,model=build_graph(settings,db);model.capture_io=True
before=json.loads(db.latest_report('danang4')['payload_json'])['report']
executions=len(db.list_tool_executions('revision:danang4',limit=10000))
query='请把餐饮推荐去重，合并同一家店的不同名字，其他安排保留。'
graph.invoke({'messages':[HumanMessage(content=query)],'current_query':query,'user_id':'revision','trip_id':'danang4','thread_id':'revision:danang4'},{'configurable':{'thread_id':'revision:danang4'}})
after=json.loads(db.latest_report('danang4')['payload_json'])['report']
ids=[f['id'] for f in after['food'] if f.get('id')]
checks={'unique_food_ids':len(ids)==len(set(ids)),'kept_transport':before['transport']==after['transport'],'kept_hotels':before['hotel_regions']==after['hotel_regions'],'kept_insights':before['insights']==after['insights'],'kept_budget':before['budget']==after['budget'],'no_tool_call_needed':len(db.list_tool_executions('revision:danang4',limit=10000))==executions,'quality':after['quality']['passed'],'real_model':model.call_count>0 and model.fallback_count==0}
summary={'checks':checks,'passed':all(checks.values()),'model_calls':model.call_count,'last_report':db.latest_report('danang4')['html_path']}
(root/'food-dedupe-acceptance.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(root/'food-dedupe-model-trace.json').write_text(json.dumps(_trace_redact(model.attempt_trace),ensure_ascii=False,default=str,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2));db.close()
assert summary['passed']
