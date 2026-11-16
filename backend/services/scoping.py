"""Single choke point for reading transactions.

SECURITY RULE: route code must NEVER query Transaction/Prediction directly
(`Transaction.query`, `db.session.query/get/execute`). It must go through the helpers below,
which always add `WHERE transactions.user_id = <authenticated user>` in SQL.
Predictions are only reached through an owned transaction (join / relationship).
tests/test_security.py enforces this rule by scanning the route source code.
"""
from flask import abort
from flask_login import current_user
from ..models import Transaction


def owned_transactions():
    """Query of ONLY the authenticated user's transactions (ownership enforced in the database query)."""
    if not current_user.is_authenticated:
        abort(401)
    return Transaction.query.filter(Transaction.user_id == current_user.id)


def get_owned_transaction(tx_id):
    """The user's transaction, or None. Nonexistent and other-user IDs are indistinguishable by design."""
    return owned_transactions().filter(Transaction.id == tx_id).first()
