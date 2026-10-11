"""Root decision + actual immutable evidence files -> one auditable full choice input."""
import argparse,json,os
from pathlib import Path
# This is CPU-only; initialize Torch evidence reader with bounded threads.
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from p1_contract import ARMS,sha,validate_selection,verify_selection_evidence

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=('J5','J20'))
    p.add_argument('--stage',choices=('confirmation_recipe','full'),default='full')
    p.add_argument('--confirmation',type=Path)
    for name in ('checkpoint','config','identity','paired-report','training-source'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--original-initial-free-bytes',type=int,required=True);p.add_argument('--reason',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    paths={'checkpoint':args.checkpoint,'config':args.config,'identity':args.identity,
        'paired_report':args.paired_report,'training_source':args.training_source}
    evidence={name:dict(path=str(path.resolve()),sha256=sha(path)) for name,path in paths.items()}
    identity=json.loads(args.identity.read_text());paired=json.loads(args.paired_report.read_text())
    chosen=dict(arm=args.arm,epoch=20,checkpoint_role='last',role='last',checkpoint_sha256=evidence['checkpoint']['sha256'],
        config_sha256=evidence['config']['sha256'])
    value=dict(schema='aictrack1.p1.full_selection.v1',
        status='selected_for_full_training' if args.stage=='full' else 'selected_recipe_for_confirmation',selected_arm=args.arm,
        head_learning_rate=ARMS[args.arm],checkpoint_role='last',checkpoint_epoch=20,training_scope='fixed_split_validation',
        train_count=identity.get('training_count',identity.get('train_count')),validation_count=identity['validation_count'],
        training_complete=True,decision_basis='current_stage_train_derived_validation',used_test_or_online_scores=False,
        training_source_manifest_sha256=evidence['training_source']['sha256'],
        diagnostic_source_manifest_sha256=paired['diagnostic_source_manifest_sha256'],
        selection_protocol_sha256=paired['selection_protocol_sha256'],selection_reason=args.reason,
        original_initial_free_bytes=args.original_initial_free_bytes,evidence=evidence,
        selection_split_sha256=identity['split_sha256'],selection_rule='fixed_final20',test_access=False,
        selected_fold_checkpoint_sha256=evidence['checkpoint']['sha256'],selected_fold_config_sha256=evidence['config']['sha256'],
        candidates=[chosen],stable_gain_claim=False,formal_test_frozen=False)
    if args.stage=='full':
        if not args.confirmation:p.error('--confirmation immutable fresh matched PASS receipt is required for full')
        value['confirmation_receipt']=dict(path=str(args.confirmation.resolve()),sha256=sha(args.confirmation))
        value['confirmation_receipt_sha256']=value['confirmation_receipt']['sha256']
    elif args.confirmation:p.error('confirmation recipe cannot claim a completed fresh confirmation')
    validate_selection(value,require_confirmation=args.stage=='full')
    verify_selection_evidence(value,require_confirmation=args.stage=='full')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # Decisions are immutable inputs; do not silently replace a prior choice.
    with args.output.open('x',encoding='utf-8') as handle:handle.write(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status='selection_written',path=str(args.output.resolve()),sha256=sha(args.output),selected_arm=args.arm)))
    return 0
if __name__=='__main__':raise SystemExit(main())
