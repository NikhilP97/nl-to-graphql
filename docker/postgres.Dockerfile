FROM postgres:alpine

ENV POSTGRES_DB="blog_db"
ENV POSTGRES_USER="root_user"
ENV POSTGRES_PASSWORD="aksld2124"

COPY ./data/postgres/init/ /docker-entrypoint-initdb.d/
