-- using 1 as a seed to the RNG


select p_brand,
       p_type,
       p_size,
       count(distinct ps_suppkey) as supplier_cnt
from partsupp,
     part
where p_partkey = ps_partkey
  and p_brand <> 'Brand#41'
  and p_type not like 'MEDIUM BURNISHED%'
  and p_size in (4, 22, 35, 31, 47, 44, 30, 11)
  and ps_suppkey not in (select s_suppkey
                         from supplier
                         where s_comment like '%Customer%Complaints%')
group by p_brand,
         p_type,
         p_size
order by supplier_cnt desc,
         p_brand,
         p_type,
         p_size;
limit
-1;
