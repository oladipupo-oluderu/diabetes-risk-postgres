"""One command to run everything:  python src/pipeline.py"""
import analysis
import build_db
import eda
import train
import tune

if __name__ == "__main__":
    build_db.build_database()   # CSV -> PostgreSQL
    eda.main()                  # plots from SQL views
    train.main()                # baseline models
    tune.main()                 # tuning, threshold, interpretation (final model)
    analysis.main()             # SQL analysis queries
