# table -> list of neighbors.  Ref: journal version of Leis et al.
JOIN_ORDER_BENCHMARK_JOIN_GRAPH = {
    'aka_title': ['title'],
    'char_name': ['cast_info'],
    'role_type': ['cast_info'],
    'comp_cast_type': ['complete_cast'],
    'movie_link': ['title', 'link_type'] + [
        'complete_cast', 'aka_title', 'movie_link', 'cast_info',
        'movie_companies', 'movie_keyword', 'movie_info_idx', 'movie_info',
        'kind_type'
    ],  # movie_link.id linked to title.id which are both primary keys
    'link_type': ['movie_link'],
    'cast_info': ['char_name', 'role_type', 'title', 'aka_name'],
    'complete_cast': ['comp_cast_type', 'title'],
    'title': [
        'complete_cast', 'aka_title', 'movie_link', 'cast_info',
        'movie_companies', 'movie_keyword', 'movie_info_idx', 'movie_info',
        'kind_type'
    ],
    'aka_name': ['cast_info', 'name'],
    'movie_companies': ['title', 'company_name', 'company_type'],
    'kind_type': ['title'],
    'name': ['aka_name', 'person_info'] +
            ['cast_info'],  # name.id linked to aka_name.id which are both primary keys
    'company_type': ['movie_companies'],
    'movie_keyword': ['title', 'keyword'],
    'movie_info': ['title', 'info_type'],
    'person_info': ['name', 'info_type'],
    'info_type': ['movie_info', 'person_info', 'movie_info_idx'],
    'company_name': ['movie_companies'],
    'keyword': ['movie_keyword'],
    'movie_info_idx': ['title', 'info_type'],
}
