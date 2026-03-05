def score_candidate(resume_text, skills):

    score = 0

    for skill in skills:

        if skill.lower() in resume_text:
            score += 1

    final_score = (score / len(skills)) * 10

    return round(final_score,2)