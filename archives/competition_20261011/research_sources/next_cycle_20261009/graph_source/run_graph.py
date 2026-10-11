"""CPU-only four-candidate graph audit or one frozen-candidate confirmation."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse,json,sys,time
from pathlib import Path
from p3_contract import ROOT,sha,read,verify_source,load_assets,order_sha

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def run(a):
    import numpy as np
    import torch
    from p3_graph import GRID,MARGIN,blocked_topk,mutual_graph,smooth,p7,check
    from p3_metrics import paired
    from p3_budget import BudgetLease
    verify_source();torch.set_num_threads(2)
    if a.max_wall_seconds<=0 or a.max_wall_seconds>7200:raise ValueError('P3 shared wall limit is at most7200s')
    output=a.output.resolve()
    if output.exists():raise FileExistsError('immutable graph result already exists')
    with BudgetLease(a.budget_ledger) as budget:
        deadline=time.monotonic()+min(a.max_wall_seconds,budget.remaining)
        data=load_assets(a.assets,a.assets_sha256);check(deadline)
        grid=list(GRID);choice=None
        if a.stage=='confirm':
            if not a.choice or not a.choice_sha256 or sha(a.choice)!=a.choice_sha256:raise ValueError('immutable selected graph receipt required')
            choice=read(a.choice);point=(choice['k'],choice['lambda'])
            if (choice.get('schema')!='aictrack1.p3.graph_choice.v1' or choice.get('status')!='selected_for_confirmation'
                    or choice.get('graph_protocol_sha256')!=sha(ROOT/'graph_protocol.json')
                    or choice.get('graph_source_manifest_sha256')!=sha(ROOT/'source_manifest.json') or point not in GRID
                    or choice.get('test_access') is not False or choice.get('selection_uses_test_or_online_scores') is not False
                    or data['identity']['split_sha256']=='0c902ea38c04581eb33ef4b26fac2d37a09abb8a4fad4ae0893b89e989ac584c'
                    or data['identity']['split_sha256']!= '57189c0bd5bf195fc9d582663eb1b31282448e82ab99ca068c76c53097a658b7'):
                raise ValueError('confirmation requires one selected point on the independent fresh split')
            for name in ('result_report','result_manifest'):
                item=choice['evidence'][name]
                if sha(item['path'])!=item['sha256'] or read(item['path']).get('complete') is not True:
                    raise ValueError('selected graph experiment was not completely delivered')
            selected_report=read(choice['evidence']['result_report']['path'])
            if (selected_report.get('selected_k'),selected_report.get('selected_lambda'))!=point or selected_report.get('primary_p7_delta_pp',0)<=0:
                raise ValueError('selected point is not the positive completed experiment')
            grid=[point]
        elif a.choice or a.choice_sha256:raise ValueError('exploration never accepts a preselected external point')
        output.mkdir(parents=True);records=[];work=[];saved={}
        report=dict(schema='aictrack1.p3.graph_audit.v1',stage=a.stage,test_access=False,complete=False,
            graph_source_manifest_sha256=sha(ROOT/'source_manifest.json'),graph_protocol_sha256=sha(ROOT/'graph_protocol.json'),
            input_assets_sha256=a.assets_sha256,input_identity=data['identity'],precision=data['precision'],
            actual_runtime=dict(torch_num_threads=torch.get_num_threads(),OMP_NUM_THREADS=os.environ['OMP_NUM_THREADS'],
                MKL_NUM_THREADS=os.environ['MKL_NUM_THREADS'],OPENBLAS_NUM_THREADS=os.environ['OPENBLAS_NUM_THREADS']),
            evaluated_grid=[dict(k=k,lambda_value=l) for k,l in grid],command=sys.argv,protocols=[])
        try:
            for protocol in data['protocols']:
                check(deadline);idx=protocol['indices'];columns=protocol['columns']
                scores=data['scores'][idx][:,columns].copy();features=data['features'][idx].copy()
                neighbors,weights,neighbor_info=blocked_topk(features,20,deadline=deadline)
                graph={k:mutual_graph(neighbors,weights,k) for k in sorted(set(k for k,_ in grid))}
                raw=scores.argmax(1);base_p7,p7_info=p7(scores,deadline=deadline)
                top=np.partition(scores,-2,axis=1)[:,-2:];protected=top.max(1)-top.min(1)>=MARGIN
                name=protocol['name'];saved[name+'_base_raw']=raw;saved[name+'_base_p7']=base_p7
                summary=dict(name=name,query_count=len(idx),class_columns=columns.tolist(),
                    sample_ids_order_sha256=order_sha(protocol['ids']),source_rows=idx.tolist(),neighbor_info=neighbor_info,
                    baseline_p7_info=p7_info,candidates=[],baseline_raw=paired(raw,raw,protocol['targets'],protocol['groups'],protocol['train_counts'],protected),
                    baseline_p7=paired(base_p7,base_p7,protocol['targets'],protocol['groups'],protocol['train_counts'],protected))
                for k,l in grid:
                    transition,info,degree=graph[k];transformed,smoothing=smooth(scores,transition,degree,l,deadline=deadline)
                    raw_pred=transformed.argmax(1);p7_pred,decoder=p7(transformed,deadline=deadline)
                    raw_metrics=paired(raw_pred,raw,protocol['targets'],protocol['groups'],protocol['train_counts'],protected)
                    p7_metrics=paired(p7_pred,base_p7,protocol['targets'],protocol['groups'],protocol['train_counts'],protected)
                    entry=dict(k=k,lambda_value=l,graph=info,smoothing=smoothing,p7_decoder=decoder,raw=raw_metrics,p7=p7_metrics)
                    summary['candidates'].append(entry);saved[f'{name}_k{k}_l{l}_raw']=raw_pred;saved[f'{name}_k{k}_l{l}_p7']=p7_pred
                    if name=='balanced_supported_unique':records.append((p7_metrics['observed_accuracy'],l,k,p7_metrics['delta_pp']))
                    del transformed;check(deadline)
                report['protocols'].append(summary);work.append((protocol,scores,graph))
            winner=sorted(records,key=lambda x:(-x[0],x[1],x[2]))[0];selected=(winner[2],winner[1])
            if a.stage=='confirm' and selected!=(choice['k'],choice['lambda']):raise RuntimeError('confirmation silently changed graph point')
            combinations=[]
            for protocol,scores,graph in work:
                k,l=selected;transition,info,degree=graph[k]
                transformed,details=smooth(scores,transition,degree,l,deadline=deadline)
                name=protocol['name'];np.save(output/(name+'_selected_scores.npy'),transformed,allow_pickle=False)
                if not np.array_equal(transformed.argmax(1),saved[f'{name}_k{k}_l{l}_raw']):raise RuntimeError('selected score reconstruction differs')
                combinations.append(dict(protocol=name,points=['base/raw','base/P7','chosen_graph/raw','chosen_graph/P7'],
                    reused_candidate_predictions=True,selected_k=k,selected_lambda=l))
                del transformed
            np.savez_compressed(output/'predictions.npz',**saved)
            report.update(complete=True,selected_k=selected[0],selected_lambda=selected[1],primary_p7_delta_pp=winner[3],
                combinations_after_selection=combinations,stable_gain_claim=False,
                selection_status='selected_for_confirmation' if winner[3]>0 else 'no_positive_graph_candidate',
                confirmed_primary_observed_gain=(winner[3]>0) if a.stage=='confirm' else None,
                automatic_deployment_authorized=False,root_high_margin_and_tail_risk_review_required=True)
            choice_value=None
            if a.stage=='explore':
                value=dict(schema='aictrack1.p3.graph_choice.v1',status=report['selection_status'],k=selected[0],lambda_value=selected[1],
                    graph_protocol_sha256=report['graph_protocol_sha256'],graph_source_manifest_sha256=report['graph_source_manifest_sha256'],
                    input_assets_sha256=a.assets_sha256,checkpoint_sha256=data['identity']['checkpoint_sha256'],
                    experiment_config_sha256=data['identity']['experiment_config_sha256'],split_sha256=data['identity']['split_sha256'],
                    sample_ids_order_sha256=data['identity']['sample_ids_order_sha256'],primary_p7_delta_pp=winner[3],
                    selection_uses_test_or_online_scores=False,test_access=False,stable_gain_claim=False)
                # External field name matches the fixed public graph point, not argparse spelling.
                value['lambda']=value.pop('lambda_value');choice_value=value
            check(deadline)
        except BaseException as exc:
            report['error']=str(exc);report['status']='incomplete_no_choice_for_deployment'
            write(output/'partial_report.json',report);raise
        report['aggregate_remaining_before_run_seconds']=budget.remaining
        report['wall_seconds_before_receipt']=time.monotonic()-budget.started
        write(output/'report.json',report)
        files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in output.iterdir() if p.is_file()}
        write(output/'result_manifest.json',dict(schema='aictrack1.p3.graph_result.v1',complete=True,files=files,
            source_manifest_sha256=sha(ROOT/'source_manifest.json'),test_access=False))
        if choice_value is not None:
            choice_value['evidence']={name:dict(path=str(output/file),sha256=sha(output/file))
                for name,file in [('result_report','report.json'),('result_manifest','result_manifest.json')]}
            write(output/'graph_choice.json',choice_value)
        print(json.dumps(dict(status=report['selection_status'],selected=selected,primary_p7_delta_pp=winner[3],
            report_path=str(output/'report.json'),report_sha256=sha(output/'report.json'))));return 0

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--assets-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=('explore','confirm'),default='explore')
    p.add_argument('--choice',type=Path);p.add_argument('--choice-sha256');p.add_argument('--max-wall-seconds',type=float,default=7200.)
    p.add_argument('--budget-ledger',type=Path,default=ROOT/'artifacts/graph_budget.json');return run(p.parse_args())
if __name__=='__main__':raise SystemExit(main())
