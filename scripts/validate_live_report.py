"""Acceptance checks on reports produced by live_report_acceptance.py."""
from pathlib import Path
import json
import sys

case=sys.argv[1]
root=Path('output/report-acceptance')
result=json.loads((root/f'{case}-result.json').read_text())
plan=result['plan']
r=plan['report']
tools=[x.get('payload', x) for x in json.loads((root/f'{case}-tool-trace.json').read_text())]
model=json.loads((root/f'{case}-model-trace.json').read_text())
checks={
 'real_model':bool(model) and all(x['status']=='success' for x in model),
 'real_tools_only':bool(tools) and all(x['result']['source_type']=='mcp' for x in tools),
 'all_days':len(r['days'])==r['trip']['duration_days'],
 'expected_duration':r['trip']['duration_days']==(4 if case=='chengdu4' else 7),
 'expected_budget':r['trip']['budget']['amount']==(8000 if case=='chengdu4' else 10000),
 'expected_dates':r['trip']['travel_dates']['end_date']==('2026-11-27' if case=='chengdu4' else '2026-11-29'),
 'two_to_three_stops':all(2<=len(d['stops'])<=3 for d in r['days']),
 'unique_stops':len({s['id'] for d in r['days'] for s in d['stops']})==r['point_count'],
 'full_note_bodies':len(r['notes'])>=4 and all(n['content'].strip() for n in r['notes']),
 'note_provenance':all(s['note_ids'] for d in r['days'] for s in d['stops']),
 'main_poi_targets':all(not (('-' in s.get('map_name','') and not s['map_name'].endswith(s['name'])) or (s.get('map_name','').endswith(('大街','路','街道')) and not s['name'].endswith(('大街','路','街道')))) for d in r['days'] for s in d['stops']),
 'all_routes_verified':all(l['verified'] for d in r['days'] for l in d['legs']),
 'reasonable_local_transfers':all(l['minutes'] is not None and l['minutes']<=90 for d in r['days'] for l in d['legs']),
 'two_hotels_per_stay':all(1<=len(g['hotels'])<=2 for g in r['hotel_regions'] if g.get('nights')),
 'one_budget_baseline':all(sum(h['baseline'] for h in g['hotels'])==1 for g in r['hotel_regions'] if g.get('nights')),
 'unique_hotels':all(len({h['url'].split('?')[0] for h in g['hotels']})==len(g['hotels']) for g in r['hotel_regions']),
 'hotel_links':all(h['url'].startswith('https://') and h['pros'] and h['cons'] for g in r['hotel_regions'] for h in g['hotels']),
 'budget_honesty':r['budget']['finalized'] or r['budget']['total'] is None,
 'no_mock_content':'示例数据' not in json.dumps(r,ensure_ascii=False),
}
if case=='jiangnan7':
 checks['two_cities']=len(set(d['city'] for d in r['days']))==2
 checks['no_flight_call']=not any('flight' in x['tool_name'] for x in tools)
 checks['rail_called']=any('get-tickets' in x['tool_name'] for x in tools)
 checks['rail_segments_complete']={('上海','苏州'),('苏州','杭州'),('杭州','上海')} <= {(t['origin'],t['destination']) for t in r['transport']}
 checks['real_rail_candidates']=all(t['options'] for t in r['transport'])
 checks['rail_station_alignment']=all(t['origin'] in o.get('from_station','') and t['destination'] in o.get('to_station','') for t in r['transport'] for o in t['options'])
 checks['return_train_after_sightseeing']=all(o.get('start_time','')>='16:00' for t in r['transport'] if t['destination']=='上海' for o in t['options'])
else:
 checks['short_trip_single_city']=len(set(d['city'] for d in r['days']))==1
 from travel_agent.reports.planning import km
 points=[s for d in r['days'] for s in d['stops']]
 checks['short_trip_compact']=any(all(km(a,b)<=30 for b in points) for a in points)
path=max((root/'reports'/case).glob('*.html'), key=lambda p:p.stat().st_mtime)
html=path.read_text()
checks['rendered_html']='{{' not in html and '{%' not in html and 'ROAM' in html
summary={'case':case,'checks':checks,'passed':all(checks.values()),'tool_calls':len(tools),'model_calls':len(model),'notes':len(r['notes']),'stops':[len(d['stops']) for d in r['days']],'routes':sum(len(d['legs']) for d in r['days']),'hotels':[len(g['hotels']) for g in r['hotel_regions']],'html':str(path)}
(root/f'{case}-acceptance.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
sys.exit(0 if summary['passed'] else 1)
