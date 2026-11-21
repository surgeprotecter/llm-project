"""Live hotel-source checks independent of model availability; not an end-to-end substitute."""
from pathlib import Path
from datetime import date
import json
from travel_agent.config import Settings
from travel_agent.mcp import MCPServerManager,load_mcp_configs
from travel_agent.tools.hotel_quote import HotelQuoteTool
from travel_agent.reports.quotes import visible_hotel_quote
from travel_agent.cli import _trace_redact

settings=Settings.from_env()
manager=MCPServerManager([c for c in load_mcp_configs(settings.mcp_config_path) if c.name=='booking'])
root=Path('output/revision-acceptance/source-checks');root.mkdir(parents=True,exist_ok=True)
checks=[]
for city,first,last,country in [('Suzhou, China','2026-11-23','2026-11-26','cn'),('Hangzhou, China','2026-11-26','2026-11-29','cn')]:
 result=manager.call_tool('booking','booking_search',{'destination':city,'checkIn':first,'checkOut':last,'adults':2,'rooms':1,'maxResults':8})
 hotels=[p for p in result.data.get('properties',[]) if '/hotel/'+country+'/' in p.get('url','')]
 assert hotels,city+' returned no verified-country hotels'
 hotel=hotels[0]
 print('SEARCH',city,hotel['name'],hotel.get('location'),flush=True)
 args={'url':hotel['url'],'name':hotel['name'],'destination':city,'checkin':first,'checkout':last,'adults':2,'rooms':1,'currency':'CNY'}
 evidence=HotelQuoteTool().run(args)
 quote=visible_hotel_quote(evidence.data,(date.fromisoformat(last)-date.fromisoformat(first)).days,2,1,'CNY',first,last)
 payload={'search':result.data,'read_quote':evidence.model_dump(mode='json'),'interpreted':quote,'passed':bool(result.success and quote)}
 (root/(city.split(',')[0].lower()+'-hotel.json')).write_text(json.dumps(_trace_redact(payload),ensure_ascii=False,indent=2))
 print('QUOTE',city,quote and quote['total_price'],quote and quote['price_note'],flush=True)
 checks.append(payload['passed'])
assert all(checks),'One or more actual quote checks failed'
