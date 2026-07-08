import pytest
import os
from src.vector_store import delete_resume_from_vectorstore

def test_delete_resume_non_existent():
    """Verify that deleting a resume when no database exists returns gracefully without error."""
    # Ensure database folder doesn't exist
    if os.path.exists("./non_existent_vectorstore"):
        import shutil
        shutil.rmtree("./non_existent_vectorstore")
        
    try:
        delete_resume_from_vectorstore("dummy_non_existent.pdf")
        # Should complete without error
        assert True
    except Exception as e:
        pytest.fail(f"delete_resume_from_vectorstore raised an error unexpectedly: {e}")
