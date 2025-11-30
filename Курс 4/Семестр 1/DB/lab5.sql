-- 1 --
select s.snum, s.name, s.city, s.comm, o.onum, o.pnum, o.cnum, o.amt
from sal as s
left join ord as o on s.snum = o.snum
order by snum;

-- 2 --
select o.onum, o.amt, c.name, c.city
from ord as o
inner join cust as c on o.cnum = c.cnum
where c.city <> 'Moscow';

-- 3 --
select s.name, min(o.amt) from sal as s
left join ord as o on s.snum = o.snum and s.comm < 0.15
group by s.name;

-- 4 --
select distinct p.pnum, c.cnum
from prod as p
inner join cust as c on p.city = c.city
group by city;

-- 5 --
select * from ord
natural join sal;

-- 6 --
select distinct right(name, 1) as last_char
from cust
union
select right(city, 1)
from cust;

-- 7 --
(select city from sal
intersect
select city from cust)
except
select city from prod;

-- 8 --
select distinct s.name
from sal as s
inner join ord as o on s.snum = o.snum
where o.pnum in (
    select pnum from ord
    group by pnum
    having count(*) > 5
);
