"""add_result_provenance_columns

Revision ID: a0aa37ea19f2
Revises: dc24ead48293
Create Date: 2026-10-03 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a0aa37ea19f2'
down_revision = 'dc24ead48293'
branch_labels = None
depends_on = None


NEW_COLUMNS = [
    ('run_id', sa.String()),
    ('execution_success', sa.Boolean()),
    ('data_source', sa.String()),
    ('random_seed', sa.Integer()),
    ('provenance', sa.JSON()),
    ('validation_status', sa.String()),
    ('validation_detail', sa.JSON()),
    ('cost_usd', sa.Float()),
    ('error_message', sa.Text()),
]


def upgrade() -> None:
    """
    Add execution, validation, provenance and cost columns to results.

    Backfills execution_success and data_source from the data JSON of
    existing rows, where the director stored them before these columns existed.
    """
    for name, type_ in NEW_COLUMNS:
        op.add_column('results', sa.Column(name, type_, nullable=True))

    op.create_index('idx_results_run_id', 'results', ['run_id'], unique=False)

    results = sa.table(
        'results',
        sa.column('id', sa.String()),
        sa.column('data', sa.JSON()),
        sa.column('execution_success', sa.Boolean()),
        sa.column('data_source', sa.String()),
    )
    connection = op.get_bind()
    rows = connection.execute(sa.select(results.c.id, results.c.data)).fetchall()
    for row_id, data in rows:
        if not isinstance(data, dict):
            continue
        values = {}
        if isinstance(data.get('execution_success'), bool):
            values['execution_success'] = data['execution_success']
        if isinstance(data.get('data_source'), str):
            values['data_source'] = data['data_source']
        if values:
            connection.execute(
                sa.update(results).where(results.c.id == row_id).values(**values)
            )


def downgrade() -> None:
    """Remove the result provenance columns."""
    op.drop_index('idx_results_run_id', table_name='results')
    with op.batch_alter_table('results') as batch_op:
        for name, _ in reversed(NEW_COLUMNS):
            batch_op.drop_column(name)
