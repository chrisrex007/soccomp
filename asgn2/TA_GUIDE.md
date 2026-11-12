Gmail	Krish Charniya <krishcharniya007@gmail.com>
Doubt regarding Assignment 2 set 3 -- Social Comuting
3 messages
Krish Charniya <krishcharniya007@gmail.com>	Sat, Nov 7, 2026 at 4:02 PM
To: sagnik basu <sagnikbasu19@gmail.com>
Dear sir

I would like to clarify the following instruction in Task 2 of Assignment 2:

“After substituting \(X=A^{-1}\), the unbudgeted problem is convex. Solve unbudgeted maximisation and minimisation on ego-Facebook with projected gradient descent … from three starting points, and show that they reach the same value.”

The assignment references the KDD 2018 paper by Abebe, Kleinberg, Parkes and Tsourakakis, Opinion Dynamics with Varying Susceptibility to Persuasion [1]. However, the authors’ subsequent paper, Opinion Dynamics with Varying Susceptibility to Persuasion via Non-Convex Local Search [2], discusses a correction to the original convexity claim.

Specifically, the updated paper’s abstract states:

“We prove that the objective function is in general non-convex. Hence, formulating the problem as a convex program as in an early version of this work (Abebe et al., KDD’18) might have potential correctness issues.”

The introduction’s “Comparison with Previous Versions” paragraph also discusses this correction. Section 5, “Structural Properties of Objective Function,” explains the non-convexity and develops the structure used by the corrected optimisation approach. Professor Kleinberg’s publication page explicitly identifies this paper as superseding the KDD 2018 version [3].

Could you please clarify whether we should:

Retain the requested projected-gradient experiments, discuss the corrected theoretical result, and independently verify the obtained optimum; or
Implement the corrected paper’s optimisation method for this part?
Also, if the projected-gradient runs do not reach the same value within the numerical stopping criteria, should we report and analyse that outcome?

Thank you for your guidance.

Regards,
Krish Charniya
23CS10012

References

[1] Abebe et al. Opinion Dynamics with Varying Susceptibility to Persuasion. KDD 2018, pp. 1089–1098.
https://doi.org/10.1145/3219819.3219983

[2] Abebe et al. Opinion Dynamics with Varying Susceptibility to Persuasion via Non-Convex Local Search. See the abstract, introduction, and Section 5 of the linked preprint.
https://arxiv.org/abs/2011.02583

[3] Jon Kleinberg’s publications page, entry for the KDD 2018 paper.
https://www.cs.cornell.edu/home/kleinber/

Krish Charniya <krishcharniya007@gmail.com>	Wed, Nov 11, 2026 at 12:47 AM
To: sagnik basu <sagnikbasu19@gmail.com>
Dear Sir,

I am writing to follow up on my previous email regarding the clarification for Task 2 of Assignment 2 (Set 3).

Could you please provide guidance on how we should proceed with the experiment and optimization method?

Thank you for your time and assistance.

Best regards,
Krish Charniya
23CS10012

[Quoted text hidden]
sagnik basu <sagnikbasu19@gmail.com>	Wed, Nov 11, 2026 at 4:46 PM
To: Krish Charniya <krishcharniya007@gmail.com>
Thanks for pointing this out, Krish. Implement the corrected paper's optimisation method for this part. For part 2 of the Task 2 of the assignment, you should implement the non-convex optimisation method.

Best regards
Sagnik


[Quoted text hidden]
