from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.baidusearch import BaiduSearchTools
from agno.tools.arxiv import ArxivTools
from agno.team.team import Team
import os
from dotenv import load_dotenv


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def stunting_recommendation(
    api_key=GEMINI_API_KEY,
    age=None,
    height_cm=None,
    gender=None,
    country=None,
    disease=None,
):
    # Set up Agno Agent with Gemini model
    search_agent = Agent(
        name="Nutrition Researcher",
        role=f"You are a nutrition researcher specializing in child growth and development. Your task is to find the latest research and recommendations for preventing and treating {disease} in children.",
        model=Gemini(id="gemini-2.0-flash-exp", api_key=api_key),
        tools=[DuckDuckGoTools(), BaiduSearchTools()],
        add_name_to_instructions=True,
        instructions=f"""
            When researching {disease} in children:
            1. Focus on peer-reviewed nutrition and pediatric journals
            2. Prioritize studies published within the last 3 years
            3. Search for:
            - Causes of {disease} in children
            - Prevention strategies for {disease}
            - Treatment options for {disease} growth
            - Nutritional recommendations for optimal growth
            4. Include information specific to:
            - Age group: {age} in month with range 0-60 month
            - Gender: {gender}
            - Country: {country}
            5. Always prioritize accurate and up-to-date information
            6. Present findings in an organized manner
        """,
    )

    medical_agent = Agent(
        name="Pediatric Specialist",
        role="You are a pediatric specialist with expertise in child growth and development. Your task is to analyze the child's data and provide personalized recommendations.",
        model=Gemini(id="gemini-2.0-flash-exp", api_key=api_key),
        tools=[ArxivTools(), DuckDuckGoTools()],
        add_name_to_instructions=True,
        instructions=f"""
            Analyze the child's data and provide recommendations:
            1. Evaluate the child's growth status based on age ({age} in month, with range 0-60 month), height ({height_cm} cm), and gender ({gender})
            2. Determine if the child is at risk of or experiencing {disease}
            3. Provide specific, actionable recommendations for:
            - Dietary improvements
            - Nutritional supplements (if necessary)
            - Physical activities to promote growth
            - Regular health check-ups and monitoring
            4. Consider cultural and economic factors specific to {country}
            5. Provide a clear, easy-to-follow action plan for the child's caregivers
        """,
    )

    agent = Team(
        name="Child Growth Specialist Team",
        mode="route",
        model=Gemini(id="gemini-2.0-flash-exp", api_key=api_key),
        members=[search_agent, medical_agent],
        instructions=f"""
        You are a team of specialists working together to provide comprehensive recommendations for a child at risk of {disease}.
        1. The Nutrition Researcher will provide the latest research and general recommendations.
        2. The Pediatric Specialist will analyze the child's specific data and provide personalized advice.
        3. Combine your expertise to create a detailed, actionable plan for the child's optimal growth and development.
        4. Ensure all recommendations are age-appropriate, culturally sensitive, and feasible for implementation in {country}.
        5. Provide a summary of your findings and recommendations in a clear, easy-to-understand format for the child's caregivers.
        """,
        show_tool_calls=True,
        markdown=True,
        debug_mode=True,
        show_members_responses=True,
        enable_team_history=True,
    )

    # Analyze and provide recommendations
    prompt = f"""
    Please provide a comprehensive analysis and recommendation plan for a {age}-month-old child with range 0-60 month {gender} child in {country}, with a height of {height_cm} cm, who may be at risk of stunting.

    Include the following in your response:
    1. Growth status assessment
    2. Risk evaluation for stunting
    3. Nutritional recommendations
    4. Physical activity suggestions
    5. Medical follow-up advice
    6. Long-term growth monitoring plan
    
    Generate result with markdown format, go directly to the result without using additional sentences and with language from this country {country}
    """

    response = agent.run(prompt)
    return response.content
