"""Exercise production graph with real APIs, real MCPs and successive user turns."""
from dataclasses import replace
from pathlib import Path
import json
import re
import sys
import time

from langchain_core.messages import HumanMessage

from travel_agent.cli import _trace_redact
from travel_agent.config import Settings
from travel_agent.graph.builder import build_graph
from travel_agent.persistence import Database

case=sys.argv[1]
root=Path('output/revision-acceptance')
root.mkdir(parents=True,exist_ok=True)
queries={
    'danang4':[
        '我和同伴两人从香港出发，2026年12月10日至13日去岘港，4天3晚，每人总预算人民币8000元。帮我出旅行规划',
        '每天至少需要去两个主要景点，帮我调整一下。',
        '再帮我补充好吃的小吃和咖啡店，安排在顺路的地方。',
    ],
    'jiangnan7':[
        '请生成完整HTML旅行报告：2026年11月23日至29日，从上海出发，两人，苏州3天、杭州4天，共7天6晚，总预算10000元。每天2到3个景点，城市之间高铁，市内打车，搜索住宿和餐饮，不查机票。',
        '第二段住宿换一家评分高些的酒店，其余行程安排不变。',
    ],
}[case]
settings=replace(Settings.from_env(),tool_mode='real',require_real_model=True,database_path=root/f'{case}.sqlite3',output_dir=root)
db=Database(settings.database_path)
graph,tools,model=build_graph(settings,db)
model.capture_io=True
assert all(type(t).__name__!='MockTravelTool' for t in tools.tools.values())
original_run=tools.run
def traced_run(name,args,thread_id):
    print('TOOL_START',name,flush=True)
    result=original_run(name,args,thread_id)
    print('TOOL_END',name,result.success,result.error_type or '',flush=True)
    return result
tools.run=traced_run
thread_id='revision:'+case
print('START',case,'tools',len(tools.tools),'integration_errors',tools.integration_errors,flush=True)
started=time.monotonic()
run_dir=root/'runs'/case/time.strftime('%Y%m%d-%H%M%S')
run_dir.mkdir(parents=True,exist_ok=True)
snapshots=[]
checks={}
try:
    for index,query in enumerate(queries,1):
        before=len(db.list_tool_executions(thread_id,limit=10000))
        print('TURN',index,query,flush=True)
        result=graph.invoke({'messages':[HumanMessage(content=query)],'user_id':'revision','trip_id':case,'thread_id':thread_id,'current_query':query},
                            {'configurable':{'thread_id':thread_id},'recursion_limit':150})
        if result.get('__interrupt__'):
            raise RuntimeError('Unexpected clarification: '+str(result['__interrupt__']))
        saved=db.latest_report(case)
        assert saved, 'No report produced'
        payload=json.loads(saved['payload_json'])
        snapshots.append(payload['report'])
        report=payload['report']
        checks[f'turn{index}_destination']=('岘港' in report['trip']['destinations'][0]) if case=='danang4' else all(city in report['trip']['destinations'][0] for city in ('苏州','杭州'))
        checks[f'turn{index}_all_days']=len(report['days'])==report['trip']['duration_days']
        checks[f'turn{index}_schedule_quality']=report['quality']['passed']
        checks[f'turn{index}_two_or_three_stops']=all(2<=len(d['stops'])<=3 for d in report['days'])
        checks[f'turn{index}_correct_country']=all(s['country_code']==report['region']['country_code'] for d in report['days'] for s in d['stops'])
        checks[f'turn{index}_routes']=all(len(d['legs'])==len(d['stops'])-1 and all(l['verified'] for l in d['legs']) for d in report['days'])
        checks[f'turn{index}_hotels']=bool(report['hotel_regions']) and all(1<=len(g['hotels'])<=2 and sum(h['baseline'] for h in g['hotels'])==1 for g in report['hotel_regions'])
        checks[f'turn{index}_hotel_images']=all(h['image_url'].startswith('https://') for g in report['hotel_regions'] for h in g['hotels'])
        checks[f'turn{index}_hotel_country']=all('/hotel/'+report['region']['country_code'].lower()+'/' in h['url'] for g in report['hotel_regions'] for h in g['hotels'])
        checks[f'turn{index}_hotel_price_context']=all(any(h['price_is_total'] and h['total_price']>0 and h['nightly_price']>0 for h in g['hotels']) for g in report['hotel_regions'])
        checked_quotes=[h for g in report['hotel_regions'] for h in g['hotels'] if h['price_is_total']]
        checks[f'turn{index}_hotel_quote_version']=bool(checked_quotes) and all(h.get('quote_version')==2 for h in checked_quotes)
        totals_correct=[]
        for h in checked_quotes:
            text=' '.join(h['quote_excerpt'].split())
            current=re.findall(r'Current price '+h['currency']+r'\s*([\d,]+(?:\.\d+)?)',text,re.I)
            labelled=current or re.findall(r'(?<!Original )Price '+h['currency']+r'\s*([\d,]+(?:\.\d+)?)',text,re.I)
            totals_correct.append(not labelled or float(labelled[-1].replace(',',''))==h['total_price'])
        checks[f'turn{index}_hotel_total_not_nightly']=all(totals_correct)
        checks[f'turn{index}_body_notes']=bool(report['notes']) and all(n['content'].strip() for n in report['notes'])
        checks[f'turn{index}_insights']=bool(report['insights'])
        checks[f'turn{index}_food']=bool(report['food']) and all(f['source_url'] and f['note_ids'] for f in report['food'])
        budget=report['budget']
        checks[f'turn{index}_budget_range']=0<budget['low']<=budget['high'] and sum(i['low'] for i in budget['items'])==budget['low'] and sum(i['high'] for i in budget['items'])==budget['high']
        checks[f'turn{index}_hotel_budget']=any(i['category']=='酒店住宿' and i['low']>0 for i in budget['items'])
        html=Path(saved['html_path']).read_text()
        checks[f'turn{index}_rendered_html']='{{' not in html and '{%' not in html and '每人' in html
        if case=='danang4':
            flights=[o for t in report['transport'] for o in t['options'] if o.get('directions')]
            checks[f'turn{index}_real_round_trip']=bool(flights) and all([d['direction'] for d in f['directions']]==['outbound','return'] and all(d['origin']!=d['destination'] for d in f['directions']) for f in flights)
            checks[f'turn{index}_flight_price_basis']=all(f['quote_passengers']==1 and f['group_estimated_price']==f['per_person_price']*2 for f in flights)
            checks[f'turn{index}_no_domestic_map']=not any(s.get('coordinate_system')=='GCJ02' for d in report['days'] for s in d['stops'])
        else:
            checks[f'turn{index}_rail']=len(report['transport'])>=3 and all(t['type']=='高铁' and t['options'] for t in report['transport'])
            checks[f'turn{index}_city_stays']=[d['city'] for d in report['days']]==['苏州']*3+['杭州']*4
        if index>1:
            previous=snapshots[-2]
            checks[f'turn{index}_kept_transport']=report['transport']==previous['transport']
            checks[f'turn{index}_kept_insights']=report['insights']==previous['insights']
            if case=='danang4':
                checks[f'turn{index}_kept_hotels']=report['hotel_regions']==previous['hotel_regions']
            else:
                checks[f'turn{index}_kept_days']=report['days']==previous['days']
                checks[f'turn{index}_changed_hotel']=report['hotel_regions']!=previous['hotel_regions']
                checks[f'turn{index}_kept_first_stay']=report['hotel_regions'][0]==previous['hotel_regions'][0]
        if case=='danang4' and index==3:
            checks['food_revision_kept_itinerary']=[{k:v for k,v in d.items() if k!='food'} for d in report['days']]==[{k:v for k,v in d.items() if k!='food'} for d in snapshots[-2]['days']]
        (root/f'{case}-turn{index}.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
        print('TURN_DONE',index,'file',saved['html_path'],'stops',[len(d['stops']) for d in report['days']],'food',len(report['food']),'budget',(budget['low'],budget['high']),'new_tool_calls',len(db.list_tool_executions(thread_id,limit=10000))-before,'quality',report['quality'],flush=True)
    executions=db.list_tool_executions(thread_id,limit=10000)
    checks['all_live_tools']=bool(executions) and all(x['payload']['result']['source_type'] in ('mcp','api','http','cli') for x in executions)
    checks['no_model_fallback']=model.fallback_count==0
    if case=='jiangnan7':
        checks['no_flights']=not any('flights' in x['tool_name'] for x in executions)
    summary={'case':case,'passed':all(checks.values()),'checks':checks,'seconds':round(time.monotonic()-started),'model_calls':model.call_count,'tool_calls':len(executions),'last_report':db.latest_report(case)['html_path'],
             'model_input_tokens':sum(x.get('usage',{}).get('input_tokens',0) for x in model.attempt_trace),'model_output_tokens':sum(x.get('usage',{}).get('output_tokens',0) for x in model.attempt_trace),
             'local_model_cache_hits':sum(x['status']=='local_cache_hit' for x in model.attempt_trace),'run_dir':str(run_dir)}
    (root/f'{case}-acceptance.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (run_dir/'acceptance.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
finally:
    (root/f'{case}-model-trace.json').write_text(json.dumps(_trace_redact(model.attempt_trace),ensure_ascii=False,default=str,indent=2))
    (run_dir/'model-trace.json').write_text(json.dumps(_trace_redact(model.attempt_trace),ensure_ascii=False,default=str,indent=2))
    (root/f'{case}-tool-trace.json').write_text(json.dumps(_trace_redact([{k:v for k,v in x.items() if k!='payload_json'} for x in db.list_tool_executions(thread_id,limit=10000)]),ensure_ascii=False,default=str,indent=2))
    db.close()
if not summary['passed']:
    raise SystemExit(1)
