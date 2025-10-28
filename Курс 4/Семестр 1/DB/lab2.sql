set search_path to my_schema;

-- 1 --
select name from prod
where name like 's%s';

-- 2 --
select distinct city from cust
where city like '%r_'
order by city;

-- 3 --
select lower(substring(name from 2)) as name from sal
where length(name) > 5
order by name;

-- 4 --
select lower(name) || ': ' || rating || ' scores' as result from cust
where name like '%ev%';

-- 5 --
select ('2026-01-01'::date - current_date);

-- 6 --
select distinct 
to_chat(ord_date, 'DD') as day,
to_chat(ord_date, 'MM') as month,
to_chat(ord_date, 'YY') as year
from ord

-- 7 --
select name from sal
where name ~* '^[bcdfghjklmnpqrstvwxyz]{2,}';

-- 8 --
select name from sal
where name ~* '.k';
