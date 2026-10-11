"""Original strict fixed-split grouped-pool final provenance; no image access."""
def bound_full_train_rows(rows, parts, *, expected_total_count=148695,
                          expected_train_count=118958, expected_validation_count=29737):
    ids = [r['sample_id'] for r in rows]
    if (len(rows) != expected_total_count or len(set(ids)) != len(ids)
            or set(ids) != set(parts)
            or any(r.get('source_split') != 'train' for r in rows)
            or any(parts[sid] not in ('train', 'validation') for sid in ids)
            or sum(parts[sid] == 'train' for sid in ids) != expected_train_count
            or sum(parts[sid] == 'validation' for sid in ids) != expected_validation_count
            or any('partition' in r and r['partition'] != parts[r['sample_id']] for r in rows)):
        raise ValueError('canonical full manifest/fixed split binding mismatch')
    # The manifest has source_split, not the derived fold partition. Derive it
    # from the SHA-bound split without mutating original rows or any label.
    return [dict(r, partition=parts[r['sample_id']]) for r in rows
            if parts[r['sample_id']] == 'train']

def validate_grouped_final(checkpoint, metrics, experiment, *, manifest_sha256,
                           split_sha256, class_mapping_sha256, num_classes,
                           train_count, validation_count, expected_total_count,
                           protocol, selected_global_indices, selected_labels):
    epochs = int(experiment['training']['epochs'])
    history = checkpoint.get('history')
    if (checkpoint.get('training_scope') not in (None, 'fixed_split_validation')
            or metrics.get('training_scope') != 'fixed_split_validation'
            or checkpoint.get('competition_stage') != 'stage2'
            or checkpoint.get('config_sha256') != metrics.get('config_sha256')
            or checkpoint.get('train_manifest_sha256') != manifest_sha256
            or metrics.get('manifest_sha256') != manifest_sha256
            or checkpoint.get('split_sha256') != split_sha256
            or metrics.get('split_sha256') != split_sha256
            or checkpoint.get('class_mapping_sha256') != class_mapping_sha256
            or int(checkpoint.get('num_classes', -1)) != num_classes
            or int(metrics.get('train_count', -1)) != protocol['selected_count']
            or int(metrics.get('validation_count', -1)) != validation_count
            or train_count != protocol['full_count']
            or train_count + validation_count != expected_total_count
            or int(checkpoint.get('epoch', -1)) != epochs or epochs != 20
            or not isinstance(history, list) or len(history) != epochs
            or history != metrics.get('history')
            or any('validation' not in row for row in history)
            or checkpoint.get('training_pool_protocol') != protocol
            or protocol.get('test_access') is not False):
        raise ValueError('grouped final fixed-split provenance mismatch')
    c = checkpoint['confidence_weighting']
    if (c['sample_indices'].tolist() != selected_global_indices
            or c['class_ids'].tolist() != selected_labels
            or len(set(selected_global_indices)) != protocol['selected_count']
            or len(selected_labels) != protocol['selected_count']
            or int(c['epochs_updated']) != 20
            or c['seen'].tolist() != [1] * protocol['selected_count']):
        raise ValueError('exact selected global ID/label AC binding mismatch')
    for epoch, row in enumerate(history, 1):
        v = row['training_pool']
        if (row['epoch'] != epoch or v['protocol'] != protocol
                or v['physical_exposures'] != train_count
                or v['unique_ac_observations'] != protocol['selected_count']
                or v['attempted_optimizer_steps'] != 1859
                or v['optimizer_steps'] < 0 or v['amp_overflow_steps'] < 0
                or v['optimizer_steps'] + v['amp_overflow_steps'] != 1859):
            raise ValueError('complete grouped exposure/optimizer/AC history mismatch')
