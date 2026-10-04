"""
Unit tests for database models and operations.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from kosmos.db.models import Base, Hypothesis, Experiment, HypothesisStatus, ExperimentStatus
from kosmos.db import operations
from datetime import datetime


@pytest.fixture
def test_db():
    """Create test database in memory."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    yield session

    session.close()


class TestHypothesisCRUD:
    """Test hypothesis CRUD operations."""

    def test_create_hypothesis(self, test_db):
        """Test creating a hypothesis."""
        hypothesis = operations.create_hypothesis(
            session=test_db,
            id="hyp-1",
            research_question="Does X affect Y?",
            statement="X increases Y",
            rationale="Because of Z",
            domain="biology",
            novelty_score=0.8,
            testability_score=0.9,
            confidence_score=0.7
        )

        assert hypothesis.id == "hyp-1"
        assert hypothesis.statement == "X increases Y"
        assert hypothesis.status == HypothesisStatus.GENERATED
        assert hypothesis.novelty_score == 0.8

    def test_get_hypothesis(self, test_db):
        """Test retrieving a hypothesis."""
        # Create
        operations.create_hypothesis(
            session=test_db,
            id="hyp-2",
            research_question="Test question",
            statement="Test statement",
            rationale="Test rationale",
            domain="physics"
        )

        # Retrieve
        hypothesis = operations.get_hypothesis(test_db, "hyp-2")

        assert hypothesis is not None
        assert hypothesis.id == "hyp-2"
        assert hypothesis.domain == "physics"

    def test_list_hypotheses(self, test_db):
        """Test listing hypotheses."""
        # Create multiple
        for i in range(5):
            operations.create_hypothesis(
                session=test_db,
                id=f"hyp-{i}",
                research_question=f"Question {i}",
                statement=f"Statement {i}",
                rationale=f"Rationale {i}",
                domain="biology" if i % 2 == 0 else "physics"
            )

        # List all
        all_hyps = operations.list_hypotheses(test_db)
        assert len(all_hyps) == 5

        # Filter by domain
        bio_hyps = operations.list_hypotheses(test_db, domain="biology")
        assert len(bio_hyps) == 3

    def test_update_hypothesis_status(self, test_db):
        """Test updating hypothesis status."""
        # Create
        operations.create_hypothesis(
            session=test_db,
            id="hyp-3",
            research_question="Test",
            statement="Test",
            rationale="Test",
            domain="biology"
        )

        # Update status
        updated = operations.update_hypothesis_status(
            session=test_db,
            hypothesis_id="hyp-3",
            status=HypothesisStatus.TESTING
        )

        assert updated.status == HypothesisStatus.TESTING


class TestExperimentCRUD:
    """Test experiment CRUD operations."""

    def test_create_experiment(self, test_db):
        """Test creating an experiment."""
        # Create hypothesis first
        operations.create_hypothesis(
            session=test_db,
            id="hyp-10",
            research_question="Test",
            statement="Test",
            rationale="Test",
            domain="biology"
        )

        # Create experiment
        experiment = operations.create_experiment(
            session=test_db,
            id="exp-1",
            hypothesis_id="hyp-10",
            experiment_type="computational",
            description="Test experiment",
            protocol={"method": "t-test", "data": "dataset.csv"},
            domain="biology"
        )

        assert experiment.id == "exp-1"
        assert experiment.hypothesis_id == "hyp-10"
        assert experiment.status == ExperimentStatus.CREATED

    def test_update_experiment_status(self, test_db):
        """Test updating experiment status."""
        # Create hypothesis
        operations.create_hypothesis(
            session=test_db,
            id="hyp-11",
            research_question="Test",
            statement="Test",
            rationale="Test",
            domain="biology"
        )

        # Create experiment
        operations.create_experiment(
            session=test_db,
            id="exp-2",
            hypothesis_id="hyp-11",
            experiment_type="computational",
            description="Test",
            protocol={},
            domain="biology"
        )

        # Update to running
        updated = operations.update_experiment_status(
            session=test_db,
            experiment_id="exp-2",
            status=ExperimentStatus.RUNNING
        )

        assert updated.status == ExperimentStatus.RUNNING
        assert updated.started_at is not None

        # Update to completed
        updated = operations.update_experiment_status(
            session=test_db,
            experiment_id="exp-2",
            status=ExperimentStatus.COMPLETED,
            execution_time_seconds=30.5
        )

        assert updated.status == ExperimentStatus.COMPLETED
        assert updated.completed_at is not None
        assert updated.execution_time_seconds == 30.5


class TestResultCRUD:
    """Test result CRUD operations."""

    def test_create_result(self, test_db):
        """Test creating a result."""
        # Create hypothesis and experiment
        operations.create_hypothesis(
            session=test_db,
            id="hyp-20",
            research_question="Test",
            statement="Test",
            rationale="Test",
            domain="biology"
        )

        operations.create_experiment(
            session=test_db,
            id="exp-10",
            hypothesis_id="hyp-20",
            experiment_type="computational",
            description="Test",
            protocol={},
            domain="biology"
        )

        # Create result
        result = operations.create_result(
            session=test_db,
            id="res-1",
            experiment_id="exp-10",
            data={"mean": 5.2, "std": 1.1},
            statistical_tests={"t_test": {"p_value": 0.03}},
            interpretation="Significant result",
            supports_hypothesis=True,
            p_value=0.03
        )

        assert result.id == "res-1"
        assert result.experiment_id == "exp-10"
        assert result.supports_hypothesis is True
        assert result.p_value == 0.03

    def test_get_results_for_experiment(self, test_db):
        """Test retrieving results for an experiment."""
        # Setup
        operations.create_hypothesis(
            session=test_db,
            id="hyp-21",
            research_question="Test",
            statement="Test",
            rationale="Test",
            domain="biology"
        )

        operations.create_experiment(
            session=test_db,
            id="exp-11",
            hypothesis_id="hyp-21",
            experiment_type="computational",
            description="Test",
            protocol={},
            domain="biology"
        )

        # Create multiple results
        for i in range(3):
            operations.create_result(
                session=test_db,
                id=f"res-{i}",
                experiment_id="exp-11",
                data={"result": i}
            )

        # Retrieve
        results = operations.get_results_for_experiment(test_db, "exp-11")

        assert len(results) == 3


def _seed_experiment(session, exp_id="exp-30", hyp_id="hyp-30"):
    operations.create_hypothesis(
        session=session,
        id=hyp_id,
        research_question="Test",
        statement="Test",
        rationale="Test",
        domain="biology"
    )
    operations.create_experiment(
        session=session,
        id=exp_id,
        hypothesis_id=hyp_id,
        experiment_type="computational",
        description="Test",
        protocol={},
        domain="biology"
    )


class TestResultProvenanceColumns:
    """Execution, validation, provenance and cost columns on results (viability plan P2-0)."""

    def test_create_result_round_trips_new_fields(self, test_db):
        _seed_experiment(test_db)

        operations.create_result(
            session=test_db,
            id="res-30",
            experiment_id="exp-30",
            data={"p_value": 0.2},
            execution_success=False,
            data_source="file",
            run_id="r1",
            random_seed=42,
            provenance={"git_sha": "abc"},
            validation_status="unvalidated",
            validation_detail={"null_model": {"p_value": 0.5}},
            cost_usd=0.001,
            error_message="ExecutionError: exit 1",
            code="print('hi')",
        )
        test_db.expire_all()

        result = operations.get_result(test_db, "res-30")
        assert result.execution_success is False
        assert result.data_source == "file"
        assert result.run_id == "r1"
        assert result.random_seed == 42
        assert result.provenance == {"git_sha": "abc"}
        assert result.validation_status == "unvalidated"
        assert result.validation_detail == {"null_model": {"p_value": 0.5}}
        assert result.cost_usd == 0.001
        assert result.error_message == "ExecutionError: exit 1"
        assert operations.get_experiment(test_db, "exp-30").code_generated == "print('hi')"

    def test_new_fields_default_to_null(self, test_db):
        _seed_experiment(test_db)

        result = operations.create_result(
            session=test_db, id="res-31", experiment_id="exp-30", data={}
        )

        assert result.execution_success is None
        assert result.run_id is None
        assert result.validation_status is None
        assert result.cost_usd is None
        assert operations.get_experiment(test_db, "exp-30").code_generated is None

    def test_code_for_missing_experiment_raises(self, test_db):
        with pytest.raises(ValueError, match="exp-missing"):
            operations.create_result(
                session=test_db, id="res-32", experiment_id="exp-missing", data={}, code="x = 1"
            )

    def test_invalid_validation_status_raises(self, test_db):
        _seed_experiment(test_db)

        with pytest.raises(ValueError, match="validation_status"):
            operations.create_result(
                session=test_db, id="res-33", experiment_id="exp-30", data={},
                validation_status="maybe",
            )

    def test_get_results_for_run(self, test_db):
        _seed_experiment(test_db)
        operations.create_result(
            session=test_db, id="res-34", experiment_id="exp-30", data={}, run_id="r1"
        )
        operations.create_result(
            session=test_db, id="res-35", experiment_id="exp-30", data={}, run_id="r2"
        )
        operations.create_result(
            session=test_db, id="res-36", experiment_id="exp-30", data={}
        )

        results = operations.get_results_for_run(test_db, "r1")

        assert [r.id for r in results] == ["res-34"]
        assert operations.get_results_for_run(test_db, "r3") == []

    def test_update_result_validation(self, test_db):
        _seed_experiment(test_db)
        operations.create_result(
            session=test_db, id="res-37", experiment_id="exp-30", data={},
            supports_hypothesis=True, validation_status="unvalidated",
        )

        operations.update_result_validation(
            test_db, "res-37", "rejected", {"null_model": {"p_value": 0.6}}, False
        )
        test_db.expire_all()

        result = operations.get_result(test_db, "res-37")
        assert result.validation_status == "rejected"
        assert result.validation_detail == {"null_model": {"p_value": 0.6}}
        assert result.supports_hypothesis is False
        assert result.cost_usd is None

        operations.update_result_validation(test_db, "res-37", "validated", None, True, cost_usd=0.02)
        test_db.expire_all()

        result = operations.get_result(test_db, "res-37")
        assert result.validation_status == "validated"
        assert result.validation_detail is None
        assert result.supports_hypothesis is True
        assert result.cost_usd == 0.02

    def test_update_result_validation_rejects_bad_input(self, test_db):
        _seed_experiment(test_db)
        operations.create_result(session=test_db, id="res-38", experiment_id="exp-30", data={})

        with pytest.raises(ValueError, match="validation_status"):
            operations.update_result_validation(test_db, "res-38", "maybe")
        with pytest.raises(ValueError, match="res-missing"):
            operations.update_result_validation(test_db, "res-missing", "validated")


class TestResultProvenanceMigration:
    """Alembic revision a0aa37ea19f2 adds the columns and backfills them from the data JSON."""

    def test_upgrade_backfills_execution_success_and_data_source(self, tmp_path, monkeypatch):
        import json
        from pathlib import Path

        from alembic import command
        from alembic.config import Config
        from sqlalchemy import text

        from kosmos.config import reset_config

        db_url = f"sqlite:///{tmp_path / 'migrate.db'}"
        # alembic/env.py takes the URL from the Kosmos config, not from alembic.ini
        monkeypatch.setenv("DATABASE_URL", db_url)
        reset_config()
        try:
            # No ini file: env.py would run logging.config.fileConfig and disable
            # every existing logger for the rest of the test session.
            cfg = Config()
            cfg.set_main_option("script_location", str(Path(__file__).resolve().parents[3] / "alembic"))
            cfg.set_main_option("sqlalchemy.url", db_url)
            command.upgrade(cfg, "dc24ead48293")

            engine = create_engine(db_url)
            with engine.begin() as conn:
                conn.execute(text(
                    "INSERT INTO hypotheses (id, research_question, statement, rationale, domain) "
                    "VALUES ('h', 'q', 's', 'r', 'd')"
                ))
                conn.execute(text(
                    "INSERT INTO experiments (id, hypothesis_id, experiment_type, description, protocol, domain) "
                    "VALUES ('e', 'h', 'data_analysis', 'd', '{}', 'd')"
                ))
                for row_id, data in [
                    ("ok", {"execution_success": True, "data_source": "file"}),
                    ("failed", {"execution_success": False}),
                    ("legacy", {"p_value": 0.1}),
                ]:
                    conn.execute(
                        text("INSERT INTO results (id, experiment_id, data) VALUES (:id, 'e', :data)"),
                        {"id": row_id, "data": json.dumps(data)},
                    )

            command.upgrade(cfg, "head")

            with engine.connect() as conn:
                rows = dict(
                    (r[0], (r[1], r[2])) for r in conn.execute(
                        text("SELECT id, execution_success, data_source FROM results")
                    )
                )
            engine.dispose()
        finally:
            reset_config()

        assert rows == {"ok": (1, "file"), "failed": (0, None), "legacy": (None, None)}
