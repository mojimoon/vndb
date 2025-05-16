-- supabase

-- CREATE TABLE vn (
--   id text NOT NULL,
--   image text,
--   c_image text,
--   olang language NOT NULL,
--   l_wikidata integer,
--   c_votecount integer NOT NULL,
--   c_rating smallint,
--   c_average smallint,
--   length smallint NOT NULL,
--   devstatus smallint NOT NULL,
--   alias text NOT NULL,
--   l_renai text NOT NULL,
--   description text NOT NULL,
--   PRIMARY KEY(id)
-- );

create table vn (
    id text not null,
    c_votecount integer not null,
    c_rating smallint,
    alias text not null,
    primary key(id)
);
