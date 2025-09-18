-- 1 --
set search_path to my_schema;

-- 2 --
create table if not exists prod (
  pnum INT primary key,
  name VARCHAR(20) not null,
  weight INT not null,
  city VARCHAR(20) not null
);

create table if not exists cust (
  cnum INT primary key,
  name VARCHAR(20) not null,
  rating INT not null,
  city VARCHAR(20) not null
);

create table if not exists sal (
  snum INT primary key,
  name VARCHAR(20) not null,
  comm numeric(7,2) not null,
  city VARCHAR(20) not null
);

create table if not exists ord (
  onum INT primary key,
  pnum INT not null,
  cnum INT not null,
  snum INT not null,
  amt INT not null,
  foreign key (pnum) references prod(pnum),
  foreign key (cnum) references cust(cnum),
  foreign key (snum) references sal(snum)
);

-- 3 --
insert into prod (pnum, name, weight, city) values
(1001, 'Monitor', 2000, 'Obninsk'),
(1002, 'Keyboard', 500, 'Yekaterinburg'),
(1003, 'Mouse', 100, 'Novosibirsk'),
(1004, 'Printer', 1500, 'Saint Petersburg'),
(1005, 'Hard drive', 300,  'Moscow'),
(1006, 'Speakers', 700, 'Novosibirsk');

insert into cust (cnum, name, rating, city) values
(2001, 'Ivanov', 100, 'Perm'),
(2002, 'Petrov', 100, 'Moscow'),
(2003, 'Vasiliev', 200, 'Yekaterinburg'),
(2004, 'Dmitriev', 200, 'Kransoyarsk'),
(2005, 'Skvortcov', 300, 'Moscow'),
(2006, 'Avdeev', 400, 'Novosibirsk'),
(2007, 'Smirnov', 100, 'Omsk');

insert into sal (snum, name, comm, city) values
 (3001, 'DNS', 0.11, 'Novosibirsk'),
 (3002, 'Citylink', 0.12, 'Saint Petersburg'),
 (3003, 'MVideo', 0.15, 'Yekaterinburg'),
 (3004, 'Inline', 0.13, 'Vladivostok'),
 (3005, 'Elbrus', 0.11, 'Moscow');

 insert into ord (onum, pnum, cnum, snum, amt) values
 (4001, 1001, 2001, 3001, 2),
 (4002, 1001, 2002, 3001, 1),
 (4003, 1001, 2006, 3002, 10),
 (4004, 1001, 2003, 3003, 5),
 (4005, 1001, 2004, 3004, 5),
 (4006, 1002, 2001, 3005, 7),
 (4007, 1002, 2002, 3001, 8),
 (4008, 1003, 2001, 3002, 3),
 (4009, 1003, 2006, 3003, 1),
 (4010, 1003, 2007, 3004, 9),
 (4011, 1003, 2004, 3003, 6),
 (4012, 1004, 2002, 3001, 6),
 (4013, 1004, 2001, 3002, 4),
 (4014, 1004, 2001, 3004, 4),
 (4015, 1004, 2006, 3003, 3),
 (4016, 1004, 2004, 3005, 3),
 (4017, 1004, 2007, 3002, 2),
 (4018, 1005, 2001, 3005, 1),
 (4019, 1006, 2004, 3005, 2),
 (4020, 1006, 2003, 3002, 3);

-- 4 --
select * from prod;
select * from cust;
select * from sal;
select * from ord;

-- 5 --
insert into sal (snum, name, comm, city) values
 (3006, 'Astra', 0.16, 'Innopolis'),
 (3007, 'RedSoft', 0.13, 'Moscow');

-- 6 --
delete from sal where snum = 3007;

-- 7 --
alter table ord
add column if not exists ord_date date;

-- 8 --
update ord
set ord_date = '2025-09-01';

-- 9 --
update ord
set ord_date = '2024-12-31' where pnum = 1002;

-- 3.1 --
select * from ord
where cnum = 2002;

-- 3.2 --
select cnum, city, name, rating from cust
where rating >= 200 

-- 3.3 --
select distinct pnum from ord
where cnum <= 2005

-- 3.4 --
select * from cust
where rating < 200 and (city = 'Saint Petersburg' or city = 'Novosibirsk')

-- 3.5 --
select * from ord
where cnum = 2004 or cnum = 2005 or cnum = 2006;

select * from ord
where cnum >= 2004 and cnum <= 2006;

select * from ord
where cnum in (2004, 2005, 2006);

