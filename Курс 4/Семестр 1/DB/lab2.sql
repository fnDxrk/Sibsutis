set search_path to my_schema;

-- 1 --

select name from prod
where name like 's%s';

-- 2 --

select distinct city from cust 
where city like '%r_' 
order by city;

-- 3 --

select lower(substring(name from 2)) as m_name from sal
where length(name) > 5
order by m_name;

-- 4 --

select lower(name) || ': ' || rating || ' scores' as result from cust
where position('ev' in name) > 0;

-- 5 --

select distinct to_char(ord_date, 'DD') as day,
to_char(ord_date, 'MM') as month,
to_char(ord_date, 'YY') as year from ord
order by year, month, day;

-- 6 --

SELECT ((DATE_TRUNC('year', lab_date)::date + INTERVAL '1 year')::date - lab_date) AS days_left
FROM (SELECT DATE '2025-09-11' AS lab_date) t;

-- 7 --

select name from sal
where name ~* '^[^eyuioa]{2,}';

-- 8 --

select name from sal
where name ~* '.k';

