"""Summarize completed scan outputs without loading full image vectors."""


def completed_scans(row, clip_key, ram_key):
    scans = {}
    digest = row['sha256']
    if digest and row['image_embedding'] and row['embedding_sha256'] == digest:
        scans['openclip'] = {'model_key': row['embedding_model'],
                             'matches_current_model': row['embedding_model'] == clip_key}
    if row['a1111_metadata'] or (digest and row['a1111_sha256'] == digest):
        scans['a1111'] = {'metadata_saved': bool(row['a1111_metadata']),
                          'manually_corrected': bool(row['manually_edited'])}
    if digest and row['ram_data'] and row['ram_sha256'] == digest and row['ram_model'] not in (None, 'manual'):
        scans['ram'] = {'model': 'Recognize Anything RAM++', 'model_key': row['ram_model'],
                       'matches_current_model': row['ram_model'] == ram_key,
                       'manually_corrected': bool(row['ram_manually_edited'])}
    # OpenCLIP/RAM-only rows have placeholder analysis but no caption model.
    if row['model'] and row['analysis']:
        scans['llama'] = {'model': row['model'], 'analysis_version': row['analysis_version'],
                         'manually_corrected': bool(row['manually_edited'])}
    return scans
