from fl.db import SessionLocal, RoundMetric, GlobalModel

session = SessionLocal()

print('\n--- Round Metrics ---')
for m in session.query(RoundMetric).all():
    print(f'Round {m.round_number}: Loss={m.loss:.4f}, Accuracy={m.accuracy:.4f}, Clients={m.participating_clients}')
    
print('\n--- Stored Models ---')
for m in session.query(GlobalModel).all():
    print(f'Round {m.round_number}: Model Size = {len(m.model_weights)} bytes')  # type: ignore