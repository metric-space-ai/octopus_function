from datetime import datetime, timedelta

def get_estimated_response_at(seconds: int) -> str:
    estimated_response_at = str(datetime.now() + timedelta(seconds=seconds))
    estimated_response_at = estimated_response_at + "Z"
    estimated_response_at = estimated_response_at.replace(" ", "T")

    return estimated_response_at
