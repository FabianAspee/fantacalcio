class Player:
    def __init__(self, position, name, team, initial_quotation):
        super().__init__()
        self.__name__ = name
        self.__team__ = team
        self.__position__ = position
        self.__initial_quotation__ = initial_quotation
        self.__vote_series__ = []
        self.__sorted_vote_for_match_day_for_file__ = []

    def add_vote(self, vote):
        self.__vote_series__.append(vote)

    def __add_vote_sorted_list__(self, vote):
        self.__sorted_vote_for_match_day_for_file__.append(vote)

    def take_sorted_vote_for_match_day(self, svfm):
        expression = (value for value in svfm if value != -10)
        if len(self.__sorted_vote_for_match_day_for_file__) == 0:
            for pair in expression:
                self.__add_vote_sorted_list__(pair)
        else:
            for pair in expression:
                self.__add_vote_sorted_list__(pair)

    def get_sorted_vote_for_match_day(self):
        return self.__sorted_vote_for_match_day_for_file__

    def get_name(self):
        return self.__name__

    def get_vote_series(self):
        return self.__vote_series__

    def __str__(self):
        return f'{self.__name__} : {self.__position__} : {self.__team__} : {self.__initial_quotation__}'
