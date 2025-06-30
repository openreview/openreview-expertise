#!/usr/bin/env python3
"""
Standalone script to set up mock data for container tests.
Uses the shared test utilities from test_utils.py to avoid code duplication.

This script sets up the required conferences and test data that the container tests expect:
- ABC.cc conference with reviewers and submissions
- DEF.cc conference for dataset tests

Usage:
    python tests/container_tests/setup_container_test_data.py
"""

import openreview
import sys
from pathlib import Path

# Add parent directories to path to import test modules
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
project_root = tests_dir.parent

sys.path.insert(0, str(tests_dir))
sys.path.insert(0, str(project_root))

# Import shared test utilities
from tests.test_utils import TestHelpers, ConferenceBuilder




def setup_openreview_clients():
    """Initialize OpenReview clients"""
    print("Initializing OpenReview clients...")
    client_v1 = openreview.Client(
        baseurl='http://localhost:3000', 
        username='openreview.net', 
        password=TestHelpers.strong_password
    )
    client_v2 = openreview.api.OpenReviewClient(
        baseurl='http://localhost:3001', 
        username='openreview.net', 
        password=TestHelpers.strong_password
    )
    
    # Create required test users
    TestHelpers.create_user('test@mail.com', 'SomeFirstName', 'User')
    TestHelpers.create_user('test@google.com', 'SomeTest', 'User')
    
    return client_v1, client_v2






def setup_abc_conference(client_v1, client_v2):
    """Set up ABC.cc conference with reviewers, submissions, and publications - matches test_expertise_service.py"""
    print(f"Setting up conference: ABC.cc")
    builder = ConferenceBuilder(client_v1, client_v2)
    conference = builder.create_conference(
        conference_id='ABC.cc',
        post_reviewers=True,
        post_area_chairs=False,
        post_senior_area_chairs=False,
        post_submissions=True,
        post_publications=True
    )
    print(f"Successfully set up conference: ABC.cc")
    return conference


def setup_def_conference(client_v1, client_v2):
    """Set up DEF.cc conference for dataset tests - matches test_create_dataset.py"""
    print(f"Setting up conference: DEF.cc")
    builder = ConferenceBuilder(client_v1, client_v2)
    conference = builder.create_conference(
        conference_id='DEF.cc',
        post_reviewers=True,
        post_area_chairs=False,
        post_senior_area_chairs=False,
        post_submissions=True,
        post_publications=True
    )
    print(f"Successfully set up conference: DEF.cc")
    return conference


def main():
    """Main entry point"""
    try:
        print("Starting container test data setup...")
        print("=" * 50)
        
        client_v1, client_v2 = setup_openreview_clients()
        
        print("\nSetting up conferences required by container tests...")
        
        # Set up conferences required by container tests
        setup_abc_conference(client_v1, client_v2)
        setup_def_conference(client_v1, client_v2)
        
        print("\n" + "=" * 50)
        print("Container test data setup completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\nError setting up container test data: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())