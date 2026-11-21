"""Run the real graph with live provider/tool calls; no fixtures or mock tools."""
from dataclasses import replace
from pathlib import Path
import json
import sys
import time

from langchain_core.messages import HumanMessage
from travel_agent.config import Settings
from travel_agent.graph.builder import build_graph
from travel_agent.persistence import Database

root=Path('output/report-acceptance')
root.mkdir(parents=True,exist_ok=True)
case=sys.argv[1] if len(sys.argv)>1 else 'chengdu4'
queries={
 'chengdu4':'请生成完整旅行行程与HTML报告：2026年11月24日至27日，香港出发去成都，2人，4天3晚，总预算8000元人民币。读取小红书笔记正文，按片区每天安排2至3个主要景点，用高德核验距离和驾车时间，市内打车，每片区1至2家酒店并选一间预算基准，查询往返机票并分项预算。首末日留出交通时间。',
 'jiangnan7':'请生成完整旅行行程与HTML报告：2026年11月23日至29日，从上海出发，苏州与杭州7天6晚，两人，总预算10000元人民币。苏州连续3天，杭州连续4天。读取小红书攻略笔记正文，每天2至3个主要景点，按城市片区集中安排，市内打车，高德核验路线，跨城市查高铁，每城1至2个酒店候选并选预算基准，分别核算住宿和交通，不查机票。',
}
s=replace(Settings.from_env(),tool_mode='real',require_real_model=True,database_path=root/f'{case}.sqlite3',output_dir=root)
db=Database(s.database_path)
graph,tools,model=build_graph(s,db)
model.capture_io=True
assert all(type(t).__name__!='MockTravelTool' and t.spec.name not in ('search_attractions','search_hotels','calculate_budget') for t in tools.tools.values())
print('START',case,'real_tools',len(tools.tools),'errors',tools.integration_errors,flush=True)
started=time.time()
try:
 result=graph.invoke({'messages':[HumanMessage(content=queries[case])],'user_id':'acceptance','trip_id':case,'thread_id':'acceptance:'+case,'current_query':queries[case]}, {'configurable':{'thread_id':'acceptance:'+case},'recursion_limit':150})
 (root/f'{case}-result.json').write_text(json.dumps(result,ensure_ascii=False,default=str,indent=2))
 print('RESPONSE',result.get('assistant_response'),flush=True)
 print('REPORT',db.latest_report(case),flush=True) if not result.get('plan') else print('REPORT_FILE',db.latest_report(case)['html_path'],flush=True)
except Exception as exc:
 print('ERROR',type(exc).__name__,str(exc),flush=True)
 raise
finally:
 from travel_agent.cli import _trace_redact
 (root/f'{case}-model-trace.json').write_text(json.dumps(_trace_redact(model.attempt_trace),ensure_ascii=False,default=str,indent=2))
 (root/f'{case}-tool-trace.json').write_text(json.dumps(_trace_redact([{k:v for k,v in item.items() if k!='payload_json'} for item in db.list_tool_executions('acceptance:'+case, limit=10000)]),ensure_ascii=False,default=str,indent=2))
 print('END seconds',round(time.time()-started),'calls',model.call_count,'fallbacks',model.fallback_count,flush=True)
 db.close()
