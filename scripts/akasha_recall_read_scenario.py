"""Isolated #909 acceptance using actual MessageLog, no providers or graph queries."""
from __future__ import annotations
import os
os.environ['CUA_TELEMETRY_ENABLED'] = 'false'
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['OTEL_TRACES_EXPORTER'] = 'none'
os.environ['OTEL_METRICS_EXPORTER'] = 'none'
import argparse
import asyncio
import threading
from datetime import UTC, datetime, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
from uuid import uuid4
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--output', type=Path, required=True, help='New JSON evidence file; never overwritten')
parser.add_argument('--baseline-ref', help='Optional Git revision with the original Inspector, e.g. e8224559cf1eab49c991155e1f3704fbb0204c60')
args = parser.parse_args()
args.source = args.source.resolve()
if args.output.exists():
    parser.error('--output must not already exist')
sys.path.insert(0, str(args.source))
from plugins.ui.contract import PluginUiRpcInvalidRequest
from plugins.ledger.contract import CallRef, ContentPart, ContentReferences, Control, Input, Output, ToolCall, ToolResult
from plugins.tools.contract import durable_call_key
import plugins.ledger.log as storage
from plugins.ledger.log import MessageLog
from plugins.akasha.inspector import RecallInspector
from plugins.akasha.recalls import ContextSource, Hit, ProgramSource, Recall, RecallRecords, ToolSource, context_identity
from plugins.turn_projection.plugin import TurnProjection
BASE = args.baseline_ref
old_module = None
if BASE:
    old_source = subprocess.check_output(['git', 'show', BASE + ':plugins/akasha/inspector.py'], cwd=args.source, text=True)
    old_module = importlib.util.module_from_spec(importlib.util.spec_from_loader('plugins.akasha._baseline_inspector_909', loader=None))
    sys.modules[old_module.__name__] = old_module
    exec(compile(old_source, 'baseline-inspector.py', 'exec'), old_module.__dict__)
FILES = tuple('plugins/akasha/' + name for name in ('inspector.py', 'recalls.py', 'runtime.py', 'plugin.py'))
def source_hashes():
    return {name: hashlib.sha256((args.source / name).read_bytes()).hexdigest() for name in FILES}
STAMP=datetime(2026,9,1,tzinfo=UTC)
projection=TurnProjection()
summary={'source':str(args.source),'baseline_ref':BASE,
    'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source,text=True).strip(),
    'tracked_dirty':bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=args.source,text=True).strip()),
    'source_sha256':source_hashes(), 'scenario_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'telemetry_disabled_before_import':True,'checks':[],'responses':{}}
counts={}
old_owner=storage._owner_record
old_message=storage._message
def counted_owner(row):
    counts['recall_rows']=counts.get('recall_rows',0)+1
    counts['recall_bytes']=counts.get('recall_bytes',0)+len(row['value'].encode())
    return old_owner(row)
def counted_message(row):
    counts['message_rows']=counts.get('message_rows',0)+1
    return old_message(row)
storage._owner_record=counted_owner
storage._message=counted_message

def recall(origin, *, n=0, hits=()):
    return Recall(learning_binding='learning',graph_version=100,source=origin,
        timestamp=STAMP+timedelta(seconds=n),limit=40,hits=hits,active_basin_count=0,pushes=0,residual_l1=0.)
def append(log, sid, mid, body, *, source='conversation',author=None):
    author=author or ('user' if isinstance(body,Input) else 'assistant')
    return log.writer(sid,author=author,source=source,body_types=(type(body),),
        content={'text':lambda _:ContentReferences(),'akasha.recall':lambda _:ContentReferences()},
        call_ref=body.call_ref if isinstance(body,ToolResult) else None,check_call=lambda _:None).append(mid,body)
def inp(log,sid,mid,**kw): return append(log,sid,mid,Input((ContentPart('text',mid),)),**kw)
def output(log,sid,mid,finish='complete',parts=None,**kw):
    return append(log,sid,mid,Output((ContentPart('text',mid),) if parts is None else parts,finish),**kw)
def auto(records,sid,mid,seq,*,source='conversation',identity=None,n=0,hits=()):
    key=identity or context_identity(sid,source,mid)
    records.save(key,recall(ContextSource(session_id=sid,source=source,through_seq=seq),n=n,hits=hits))
    return key

def digest(log):
    with sqlite3.connect(log._path) as db:
        rows={t:db.execute(f'SELECT * FROM {t} ORDER BY rowid').fetchall() for t in ['messages','owner_records','bindings']}
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
def inspector(log,records,baseline=False):
    def listing():
        counts['list_calls']=counts.get('list_calls',0)+1
        if not baseline: raise AssertionError('recall.turn called full history list')
        return records.list()
    def read(key):
        counts['point_reads']=counts.get('point_reads',0)+1
        return records.read(key)
    def page(before):
        counts['legacy_pages']=counts.get('legacy_pages',0)+1
        return records.legacy_page(before)
    cls=old_module.RecallInspector if baseline else RecallInspector
    kw={} if baseline else {'legacy_page':page}
    return cls(read=read,list_records=listing,catalog=log.catalog(),**kw)
def query(view,log,sid,mid,*,source='',offset=0):
    before=digest(log)
    counts.clear()
    started=time.perf_counter()
    result=view.for_turn(sid,mid,source,projection,offset=offset)
    measured={**counts,'ms':round((time.perf_counter()-started)*1000,3)}
    assert digest(log)==before,'read mutated authoritative rows'
    assert len(json.dumps(result,ensure_ascii=False,separators=(',',':')).encode()) <= 192*1024
    return result,measured

def ids(result): return [r['query_id'] for r in result['items']]

with tempfile.TemporaryDirectory(prefix='issue909-') as root:
    root=Path(root)
    # Current record load; source history cold vs changed-head warm decode costs.
    for n in (398,3980):
        log=MessageLog(root/f'load-{n}.db'); records=RecallRecords(log.owner('plugin:akasha'))
        for i in range(n):
            records.save(context_identity('unrelated','conversation',str(i)),recall(
                ContextSource(session_id='unrelated',source='conversation',through_seq=i),
                hits=tuple(Hit(node_id=j,session_id='unrelated',message_ids=(f'hit-{j}',),score=.1,
                    lane='completion',sources=('x'*500,)) for j in range(10))))
        for i in range(n//2):
            inp(log,'session',f'u-{i}'); output(log,'session',f'a-{i}')
        anchor=inp(log,'session','current'); aid=auto(records,'session','current',anchor.seq)
        baseline=inspector(log,records,True) if BASE else None
        current=inspector(log,records)
        bc=bwc=None
        if baseline is not None:
            br,bc=query(baseline,log,'session','current'); assert ids(br)==[aid]
        cr,cc=query(current,log,'session','current')
        assert ids(cr)==[aid]
        repeated,rc=query(current,log,'session','current')
        log.save_binding('tool-binding',{'scenario':True})
        call=output(log,'session','call','continue',(ToolCall('tool-binding',{}),))
        if baseline is not None: _,bwc=query(baseline,log,'session','current')
        _,cwc=query(current,log,'session','current')
        if bc is not None: assert bc['recall_rows']==n+2
        assert cc['recall_rows']==rc['recall_rows']==cwc['recall_rows']==1
        assert cc.get('list_calls',0)==0 and cwc.get('list_calls',0)==0
        assert cwc['message_rows'] <= 4
        if bwc is not None: assert bwc['message_rows'] >= n
        summary['checks'].append({'case':'decode-load','unrelated':n,'baseline_cold':bc,'candidate_cold':cc,
            'candidate_unchanged':rc,'baseline_new_message':bwc,'candidate_new_message':cwc})
        log.close()
    # Legacy UUID coverage, 64-row pages, earliest selection incl deterministic coexistence.
    log=MessageLog(root/'legacy.db'); records=RecallRecords(log.owner('plugin:akasha'))
    a=inp(log,'s','A'); canonical=auto(records,'s','A',a.seq,n=1000)
    legacy_keys=[]
    for i in range(145):
        identity=f'{i%16:x}{i:031x}'
        legacy_keys.append(identity)
        auto(records,'s' if i in (0,120) else 'unrelated','A',a.seq,identity=identity,n=200-i)
    view=inspector(log,records); pages=[]
    while True:
        result,c=query(view,log,'s','A'); pages.append(c)
        if not result.get('legacy_pending'): break
        assert result['items']==[] and result['next_offset'] is None
        summary['responses']['legacy_pending']=result
    assert ids(result)==[legacy_keys[120]] and len(pages)==3
    assert pages[0]['recall_rows']==pages[1]['recall_rows']==64 and pages[2]['recall_rows']==19
    _,warm=query(view,log,'s','A'); assert warm['recall_rows']==2 and warm.get('legacy_pages',0)==0
    def retained_size(value):
        size = sys.getsizeof(value)
        if isinstance(value, dict): size += sum(retained_size(k)+retained_size(v) for k,v in value.items())
        elif isinstance(value, (tuple,list)): size += sum(map(retained_size,value))
        return size
    summary['checks'].append({'case':'legacy-bounded-earliest','rows':145,'pages':pages,'warm':warm,
        'directory_entries':sum(map(len,view._legacy.values())),'directory_stores_bodies':False,'directory_recursive_size_upper_bound':retained_size(view._legacy)})
    log.close()
    # Multi-part CallRefs, A/B, late automatic, late tool after abandon, closed/quiet, pause/failure.
    log=MessageLog(root/'semantics.db'); records=RecallRecords(log.owner('plugin:akasha'))
    log.save_binding('tool-binding',{'scenario':True})
    a=inp(log,'s','A'); view=inspector(log,records)
    result,_=query(view,log,'s','A'); assert ids(result)==[] and result['pending']
    background=inp(log,'s','background',author='service')
    b=inp(log,'s','B')
    autoa=auto(records,'s','A',a.seq)
    result,_=query(view,log,'s','A'); assert ids(result)==[autoa] and result['pending']
    summary['responses']['automatic_open']=result
    result,_=query(view,log,'s','B'); assert ids(result)==[] and result['pending']
    # A different source's Input must not steal conversation ownership.
    other=inp(log,'s','other',source='wake'); auto(records,'s','other',other.seq,source='wake')
    result,_=query(view,log,'s','B'); assert ids(result)==[]
    # Calls belong to B; two separate parts are addressed independently.
    req=output(log,'s','request','continue',(ContentPart('text','before'),ToolCall('tool-binding',{}),ToolCall('tool-binding',{})))
    refs=[CallRef('request',i) for i in (1,2)]
    c=inp(log,'s','C')
    for i,ref in enumerate(refs):
        key='tool:'+durable_call_key(ref) if i==0 else 'tool:historical-execution-key'
        records.save(key,recall(ToolSource(session_id='s',call_ref=ref),n=10+i))
        append(log,'s',f'result-{i}',ToolResult(ref,'success',(ContentPart('akasha.recall',{'retrieval_ref':key}),)))
    result,_=query(view,log,'s','B'); assert len(ids(result))==2 and result['pending']
    summary['responses']['tools_open']=result
    result,_=query(view,log,'s','C'); assert not ids(result) and result['pending']
    append(log,'s','paused',Control('pause',c.seq)); append(log,'s','failure',Control('failure',c.seq))
    result,_=query(view,log,'s','B'); assert result['pending']
    output(log,'s','complete')
    result,_=query(view,log,'s','B'); assert len(ids(result))==2 and not result['pending']
    summary['responses']['tools_closed']=result
    d=inp(log,'s','D'); req=output(log,'s','late-request','continue',(ToolCall('tool-binding',{}),))
    ref=CallRef(req.message_id,0)
    append(log,'s','abandon',Control('abandon',req.seq))
    e=inp(log,'s','E')
    result,_=query(view,log,'s','D'); assert not result['pending']
    late='tool:'+durable_call_key(ref); records.save(late,recall(ToolSource(session_id='s',call_ref=ref),n=40))
    append(log,'s','late-result',ToolResult(ref,'success',(ContentPart('akasha.recall',{'retrieval_ref':late}),)))
    result,_=query(view,log,'s','D'); assert ids(result)==[late] and not result['pending']
    result,_=query(view,log,'s','E'); assert not ids(result) and result['pending']
    output(log,'s','quiet','quiet')
    result,_=query(view,log,'s','E'); assert not result['pending']
    # Background-only Input cannot trigger a user card.
    inp(log,'s2','system',author='system')
    result,_=query(view,log,'s2','system'); assert not result['items'] and not result['pending']
    summary['checks'].append({'case':'semantic-sequence','passed':['empty-result','A-auto-after-B','background-author',
        'multiple-sources','multiple-part-index','historical-tool-marker','A-tool-after-B','pause-failure-open',
        'complete-settled','abandon-closed','late-tool-after-abandon','quiet-settled']})
    # Forged source is rejected; read has no authority to move it to another card.
    f=inp(log,'s','F'); output(log,'s','forged-call','continue',(ToolCall('tool-binding',{}),))
    ref=CallRef('forged-call',0)
    records.save('tool:forged',recall(ToolSource(session_id='other-session',call_ref=ref)))
    append(log,'s','forged-result',ToolResult(ref,'success',(ContentPart('akasha.recall',{'retrieval_ref':'tool:forged'}),)))
    try: query(view,log,'s','F')
    except ValueError as error: assert '实际工具调用' in str(error)
    else: raise AssertionError('forged source accepted')
    summary['checks'].append({'case':'source-mismatch-rejected'})
    log.close()
    # Full response pagination and explicit oversize failure, no preview membership truncation.
    log=MessageLog(root/'pages.db'); records=RecallRecords(log.owner('plugin:akasha'))
    log.save_binding('tool-binding',{'scenario':True})
    a=inp(log,'s','A')
    for i in range(900): inp(log,'hits',f'h-{i:04}',source='history')
    big_hits=(Hit(node_id=0,session_id='hits',message_ids=tuple(f'h-{i:04}' for i in range(900)),score=.5,lane='dense',sources=('direct',)),)
    autoid=auto(records,'s','A',a.seq,hits=big_hits)
    request=output(log,'s','req','continue',tuple(ToolCall('tool-binding',{}) for _ in range(4)))
    for i in range(4):
        ref=CallRef('req',i); identity='tool:'+durable_call_key(ref)
        records.save(identity,recall(ToolSource(session_id='s',call_ref=ref),n=i+1,hits=big_hits))
        append(log,'s',f'res-{i}',ToolResult(ref,'success',(ContentPart('akasha.recall',{'retrieval_ref':identity}),)))
    output(log,'s','end')
    view=inspector(log,records); offset=0; all_ids=[]; page_sizes=[]
    while True:
        result,_=query(view,log,'s','A',offset=offset)
        all_ids+=ids(result); page_sizes.append(len(json.dumps(result,ensure_ascii=False,separators=(',',':')).encode()))
        assert all(len(row['hits'][0]['messages'])==900 for row in result['items'])
        offset=result['next_offset']
        if offset is None:break
    assert len(all_ids)==len(set(all_ids))==5 and len(page_sizes)>1
    oversized=recall(ContextSource(session_id='s',source='conversation',through_seq=a.seq),hits=(
        Hit(node_id=1,session_id='hits',message_ids=tuple(f'h-{i:04}' for i in range(900)),score=.2,lane='dense',sources=('x'*200000,)),))
    records.save('oversize',oversized)
    try:view.plugin_detail('oversize')
    except PluginUiRpcInvalidRequest:pass
    else:raise AssertionError('oversized detail silently truncated')
    summary['checks'].append({'case':'response-pagination','records':5,'page_bytes':page_sizes,'members_per_record':900,'oversize':'explicit failure'})
    # Identity bytes stay compatible for punctuation and Unicode.
    for sid,source,mid in [('会话','会话:来源','输入"\\'),('s','conversation','A')]:
        expected='context:'+hashlib.sha256(json.dumps([sid,source,mid],ensure_ascii=False).encode()).hexdigest()
        assert context_identity(sid,source,mid)==expected
    summary['checks'].append({'case':'identity-exact-bytes','passed':True})
    log.close()
    # Large closed history is quiescent after durable success/denial settlement.
    log=MessageLog(root/'closed.db'); records=RecallRecords(log.owner('plugin:akasha'))
    log.save_binding('tool-binding',{'scenario':True})
    for i in range(200):
        inp(log,'s',f'closed-{i}')
        req=output(log,'s',f'closed-call-{i}','continue',(ToolCall('tool-binding',{}),))
        ref=CallRef(req.message_id,0)
        if i%2:
            append(log,'s',f'closed-abandon-{i}',Control('abandon',req.seq))
        append(log,'s',f'closed-result-{i}',ToolResult(ref,'denied' if i%2 else 'success',()))
        if not i%2: output(log,'s',f'closed-end-{i}')
    view=inspector(log,records)
    for i in range(200):
        result,_=query(view,log,'s',f'closed-{i}')
        assert not result['pending'] and not result['items']
    summary['checks'].append({'case':'settled-closed-history','cards':200,'subscribers_needed':0})
    log.close()

    # The actual Tools abandon owner commits terminal before a canceled save drains.
    async def canceled_save():
        from agent.plugin_composition.tasks import Tasks
        from plugins.akasha.application.consumer import run_memory_job
        from plugins.reply.status import ReplyState
        from plugins.tools.abandon import abandon_call, reject_start
        from plugins.tools.api import MessageReply
        from plugins.tools.execution import _fingerprint
        log=MessageLog(root/'canceled-save.db'); records=RecallRecords(log.owner('plugin:akasha'))
        log.save_binding('tool-binding',{'scenario':True})
        inp(log,'s','cancel-input')
        request=output(log,'s','cancel-call','continue',(ToolCall('tool-binding',{}),))
        ref=CallRef(request.message_id,0); effect_key=durable_call_key(ref); rid='tool:'+effect_key
        reader=log.reader('s')
        writer=log.writer('s',author='tool',source='conversation',body_types=(ToolResult,),call_ref=ref,
            content={'text':lambda _:ContentReferences()})
        reply=MessageReply('cancel-result',ref,reader,writer,reject_start)
        state=log.owner('plugin:tools')
        state.transact(lambda tx:tx.save(effect_key,{'version':1,'phase':'started','reply_id':'cancel-result',
            'request':_fingerprint('tool-binding',{},reply),'binding':'tool-binding','arguments':{}},expected_version=None))
        tasks=Tasks(); status=ReplyState(); started=threading.Event(); release=threading.Event(); opened=asyncio.Event()
        def delayed_save():
            started.set()
            assert release.wait(10), 'scenario failed to release save worker'
            records.save(rid,recall(ToolSource(session_id='s',call_ref=ref)))
        async def effect(_task): await run_memory_job(delayed_save)
        tool=await tasks.admit(('effects',effect_key),lambda slot:slot.start(effect))
        async def source(task):
            with status.open(task,'s','conversation'):
                opened.set()
                try: await tool.join()
                finally:
                    while not tool.done:
                        try: await tool.join()
                        except asyncio.CancelledError: pass
        source_task=await tasks.admit('source',lambda slot:slot.start(source))
        await opened.wait(); assert await asyncio.to_thread(started.wait,10)
        append(log,'s','cancel-abandon',Control('abandon',request.seq))
        terminal=await abandon_call(state,tasks,reply,task_key='effects')
        assert terminal.outcome=='interrupted'
        source_task.cancel()
        view=inspector(log,records); result,_=query(view,log,'s','cancel-input')
        assert not result['items'] and not result['pending']
        if BASE:
            before,_=query(inspector(log,records,True),log,'s','cancel-input')
            assert not before['items'] and not before['pending']
        activity=status.snapshot('s')
        assert len(activity)==1 and not activity[0].active
        release.set()
        await asyncio.gather(tool.join(),source_task.join(),return_exceptions=True)
        assert not status.snapshot('s')
        result,_=query(view,log,'s','cancel-input')
        assert ids(result)==[rid] and not result['pending']
        await tasks.close(); status.close(); log.close()
        summary['checks'].append({'case':'abandon-save-drain','terminal_before_save':True,
            'inactive_reply_activity_retained_until_save':True,'explicit_reread_reveals_recall':True,'automatic_post_terminal_visibility':'pre-existing gap; not fixed by this patch','baseline_same_terminal_pending':bool(BASE)})
    asyncio.run(canceled_save())

assert summary['source_sha256']==source_hashes(),'source changed during scenario'
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('x') as output_file:
    json.dump(summary,output_file,ensure_ascii=False,indent=2)
    output_file.write('\n')
print(f"Passed {len(summary['checks'])} scenario groups; evidence: {args.output}")
