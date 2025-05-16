create table ulist_vns (
    uid int4 not null,
    vid int4 not null,
    lastmod date not null,
    vote int2 not null,
    notes text,
    state int2 not null,
    norm int2 not null,
    primary key (uid, vid)
);