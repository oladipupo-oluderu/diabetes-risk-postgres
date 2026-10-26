-- One-time setup for a LOCAL PostgreSQL install (not needed with Docker Compose).
-- Run as a superuser:  psql -U postgres -f sql/00_local_setup.sql
CREATE USER diabetes_user WITH PASSWORD 'diabetes_pass';
CREATE DATABASE diabetes_db OWNER diabetes_user;
