import os
import sys
import json

# Define expected rows
EXPECTED_ROWS = [
    ('SxNx1Xf83yl', '~Harold_Rice1', '0.50'),
    ('HgbgymMI21e', '~Harold_Rice1', '0.52'),
    ('SxNx1Xf83yl', '~C.V._Lastname1', '0.49'),
    ('HgbgymMI21e', '~C.V._Lastname1', '0.46'),
    ('SxNx1Xf83yl', '~Zonia_Willms1', '0.68'),
    ('HgbgymMI21e', '~Zonia_Willms1', '0.59'),
    ('SxNx1Xf83yl', '~Royal_Toy1', '0.66'),
    ('HgbgymMI21e', '~Royal_Toy1', '0.56')
]

def verify_results(output_dir):
    # Check if files exist
    assert os.path.exists(os.path.join(output_dir, 'request.json'))
    assert os.path.exists(os.path.join(output_dir, 'job_config.json'))
    assert os.path.exists(os.path.join(output_dir, 'metadata.json'))
    assert os.path.exists(os.path.join(output_dir, 'test_container.jsonl'))
    assert os.path.exists(os.path.join(output_dir, 'test_container_sparse.jsonl'))

    # Test archive files
    assert os.path.exists(os.path.join(output_dir, 'archives'))
    for file in os.listdir(os.path.join(output_dir, 'archives')):
        assert file.endswith('.jsonl')

    # Test embedding files
    assert os.path.exists(os.path.join(output_dir, 'pub2vec_specter.jsonl'))
    assert os.path.exists(os.path.join(output_dir, 'pub2vec_scincl.jsonl'))
    assert os.path.exists(os.path.join(output_dir, 'sub2vec_scincl.jsonl'))
    assert os.path.exists(os.path.join(output_dir, 'sub2vec_specter.jsonl'))
    
    # Check real scores
    with open(os.path.join(output_dir, 'test_container.jsonl'), 'r') as f:
        lines = [json.loads(line) for line in f]
        
    for submission_id, user_id, expected_score in EXPECTED_ROWS:
        found = False
        for data in lines:
            if data['submission'] == submission_id and data['user'] == user_id:
                assert str(data['score']).startswith(expected_score), \
                    f"Expected score for {submission_id}/{user_id} to start with {expected_score}, got {data['score']}"
                found = True
                break
        assert found, f"Couldn't find expected row for {submission_id}/{user_id}"

if __name__ == '__main__':
    verify_results(sys.argv[1])
