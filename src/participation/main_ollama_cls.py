import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
load_dotenv()
print("this is a test, cn ")


def main():
    print("Hello from Python course!")
    information = """
   Guido van Rossum (Dutch: [ˈxidoː vɑn ˈrɔsʏm]; born 31 January 1956) is a Dutch programmer. He is the creator of the Python programming language, for which he was the "benevolent dictator for life" (BDFL) until he stepped down from the position on 12 July 2018.[4][5] He remained a member of the Python Steering Council through 2019, and withdrew from nominations for the 2020 election.[6]

Life and education
Van Rossum was born and raised in the Netherlands, where he received a master's degree in mathematics and computer science from the University of Amsterdam in 1982. He received a bronze medal in 1974 in the International Mathematical Olympiad.[7] His brother, Just van Rossum, is a type designer and programmer who designed the typeface used in the "Python Powered" logo.[8]

Van Rossum lives in Belmont, California, with his wife, Kim Knapp,[9] and their son.[10][11][12]

Work
Centrum Wiskunde & Informatica
While working at the Centrum Wiskunde & Informatica (CWI), Van Rossum wrote and contributed a glob() routine to BSD Unix in 1986[13][14] and helped develop the ABC programming language. He once stated, "I try to mention ABC's influence because I'm indebted to everything I learned during that project and to the people who worked on it."[15] He also created Grail, an early web browser written in Python, and engaged in discussions about the HTML standard.[16]

He has worked for various research institutes, including the Centrum Wiskunde & Informatica (CWI) in the Netherlands, the U.S. National Institute of Standards and Technology (NIST), and the Corporation for National Research Initiatives (CNRI). In May 2000, he left CNRI along with three other Python core developers to work for tech startup BeOpen.com, which subsequently collapsed by October of the same year.[17][18] From late 2000 until 2003 he worked for Zope Corporation. In 2003 Van Rossum left Zope for Elemental Security. While there he worked on a custom programming language for the organization.[19]

Google
From 2005 to December 2012, Van Rossum worked at Google, where he spent half of his time developing the Python language. At Google, he developed Mondrian, a web-based code review system written in Python and used within the company. He named the software after the Dutch painter Piet Mondrian.[20] He named Rietveld, another related software project, after Gerrit Rietveld, a Dutch designer.[21] On 7 December 2012, Van Rossum left Google.[22]

Dropbox
In January 2013, Van Rossum started working at the cloud file storage company Dropbox.[23][24]

In October 2019, Van Rossum left Dropbox and officially retired.[25][26][27]

Microsoft
On 12 November 2020 Van Rossum announced that he was coming out of retirement to join the Developer Division at Microsoft. He currently holds the title Distinguished Engineer at Microsoft.[28][29][30]

Python
In December 1989, Van Rossum had been looking for a "'hobby' programming project that would keep [him] occupied during the week around Christmas" as his office was closed when he decided to write an interpreter for a "new scripting language [he] had been thinking about lately: a descendant of ABC that would appeal to Unix/C hackers". He chose the name Python for his programming language not because of the snake type but the comedy series Monty Python's Flying Circus.

He has explained that Python's predecessor, ABC, was inspired by SETL, noting that ABC co-developer Lambert Meertens had "spent a year with the SETL group at NYU before coming up with the final ABC design".[31]

On 12 July 2018, Van Rossum announced that he would be stepping down from the position of benevolent dictator for life of the Python programming language.[32]

"Computer Programming for Everybody" proposal
In 1999, Van Rossum submitted a funding proposal to the Defense Advanced Research Projects Agency (DARPA) called "Computer Programming for Everybody", in which he further defined his goals for Python:

An easy and intuitive language just as powerful as major competitors
Open source, so anyone can contribute to its development
Code that is as understandable as plain English
Suitability for everyday tasks, allowing for short development times
In 2019, Python became the second most popular language on GitHub, the largest source code management website on the internet, after JavaScript.[33] In 2024 Python became the most used language on GitHub, overtaking JavaScript after a 10-year run as the most used language.[34] According to a programming language popularity survey[35] it is consistently among the top 10 most mentioned languages in job postings. Furthermore, Python has been among the 10 most popular programming languages every year since 2004 according to the TIOBE Programming Community Index and got the number one spot on the index in October 2021
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
# Create model
    #PLACEHOLDER#

    chain = summary_prompt_template | llm
    print("invoking chain now...")
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
