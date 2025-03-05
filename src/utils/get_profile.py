import argparse
from config import LOCAL_EVALUATION_PROFILES, SERVER_EVALUATION_PROFILES


def get_profile():
    """Parse command line arguments and return the corresponding profile

    Example:
        # Use local evaluation profile
        python demo.py --local --profile_name baseline

        # Use server evaluation profile
        python demo.py --server --profile_name vector_search
    """
    parser = argparse.ArgumentParser(description="Select evaluation profile")

    # Create mutually exclusive group to ensure only --local or --server can be selected
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--local", action="store_true", help="Use local evaluation profiles"
    )
    group.add_argument(
        "--server", action="store_true", help="Use server evaluation profiles"
    )

    parser.add_argument(
        "--profile_name", type=str, required=True, help="Profile name to use"
    )

    args = parser.parse_args()

    # Select local or server profiles
    profiles = LOCAL_EVALUATION_PROFILES if args.local else SERVER_EVALUATION_PROFILES

    # Get current profile
    for profile in profiles:
        if profile["profile_name"] == args.profile_name:
            return profile

    raise ValueError(f"Profile name '{args.profile_name}' not found")
