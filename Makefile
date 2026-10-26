up:      ; docker compose up --build --abort-on-container-exit app
test:    ; docker compose run --rm app pytest -q
psql:    ; docker compose exec db psql -U diabetes_user -d diabetes_db
down:    ; docker compose down
reset:   ; docker compose down -v
local:   ; python src/pipeline.py
