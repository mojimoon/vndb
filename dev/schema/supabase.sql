-- CREATE TABLE vn (
--   id vndbid(v) NOT NULL,
--   image vndbid(cv),
--   c_image vndbid(cv),
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

-- CREATE TABLE vn_titles (
--   id vndbid(v) NOT NULL,
--   lang language NOT NULL,
--   official boolean NOT NULL,
--   title text NOT NULL,
--   latin text,
--   PRIMARY KEY(id, lang)
-- );

-- CREATE TABLE releases (
--   id vndbid(r) NOT NULL,
--   gtin bigint NOT NULL,
--   olang language NOT NULL,
--   released integer NOT NULL, -- yyyymmdd
--   ...,
--   PRIMARY KEY(id)
-- );

-- CREATE TABLE releases_vn (
--   id vndbid(r) NOT NULL,
--   vid vndbid(v) NOT NULL,
--   rtype release_type NOT NULL,
--   PRIMARY KEY(id, vid)
-- );

-- CREATE TABLE releases_producers (
--   id vndbid(r) NOT NULL,
--   pid vndbid(p) NOT NULL,
--   developer boolean NOT NULL,
--   publisher boolean NOT NULL,
--   PRIMARY KEY(id, pid)
-- );

-- CREATE TABLE producers (
--   id vndbid(p) NOT NULL,
--   type producer_type NOT NULL,
--   lang language NOT NULL,
--   name text NOT NULL,
--   latin text,
--   alias text NOT NULL,
--   description text NOT NULL,
--   PRIMARY KEY(id)
-- );

drop table if exists vn;
create table vn (
    id int4 not null, -- vn.id
    c_votecount int4 not null, -- vn.c_votecount
    c_rating int2, -- vn.c_rating
    c_average int2, -- vn.c_average
    title_ja text, -- vn_titles[id == vndbid(v) & lang == ja].title
    title_en text, -- vn_titles[id == vndbid(v) & lang == en].title
    title_zh text, -- vn_titles[id == vndbid(v) & lang == zh-Hans].title
    alias text, -- vn.alias
    search text,
    primary key(id)
);

drop table if exists vn_titles;
create table vn_release (
    id int4 not null, -- vn.id
    rid int4 not null, -- releases_vn[id == vndbid(r) & rtype == complete]<first>.rid
    release_date date, -- releases.released
    pid int4 not null, -- releases_producers[id == vndbid(r) & developer == true]
    primary key(rid)
);
