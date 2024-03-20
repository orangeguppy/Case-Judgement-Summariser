'use client';
import axios from 'axios';
import { useEffect, useState } from "react"
import { useParams } from 'next/navigation'


export default function Model() {
    const [inputText, setInputText] = useState("");
    const [summary, setSummary] = useState("");
    const params = useParams()["model"];

    const glosarry = {
        "accused": "A person who is charged with breaking the law. Also known as a defendant.",
        "acquittal": "A decision of a judge that an accused person is not guilty or a case is not proven.",
        "adjourn": "When a case is postponed to a later date.",
        "admissibility": "Describes evidence that is accepted and allowed to be considered by the court.",
        "affidavit": "A signed statement made under oath that explains what you know or belief based on a particular set of facts.",
        "affirmation": "A declaration to tell the truth in court; one that does not involve taking a religious oath.",
        "allegation": "An accusation that has been made but not yet proven to be true.",
        "ancillary matters": "Issues regarding the children, property and maintenance in a divorce case.",
        "antecedents": "An accused person\u2019s previous criminal record.",
        "asynchronous": "Describes events that do not occur at the same time.In this case, it describes the situation where parties to the legal action will not be required to appear before the judge at the same time, or where the judge will issue orders and\n                directions without requiring the applicant to appear before them.",
        "bail": "Temporary release of an accused person awaiting trial or a hearing, usually on the condition that a sum of money is lodged to guarantee their appearance in court.",
        "bailor": "A person who is willing to provide security for the amount of money ordered by the court for the bail amount, so that an accused person may be released from remand. A bailoris also known as a \"surety\".",
        "balance of probabilities": "In civil cases, this is the standard of proof and it is the way a judge makes decisions. It means that the court is satisfied that an event occurred if, based on the evidence, the occurrence of the event was more likely than not.",
        "bond": "An undertaking in writing to perform a certain act.",
        "chambers": "The private office of a judge or judicial officer where they deal with matters outside of a court session.",
        "charge": "The crime that an accused person is thought to have committed.",
        "citation": "The letter or form that tells a witness when and where to go to court.",
        "claim trial": "When an accused person claims trial, it means they do not admit that they are guilty and wish to defendthemself against the charges at trial.",
        "codicil": "An addition or supplement that explains, modifies, or revokes a will or part of one.",
        "commissioner": "A judge, lawyer, sheriff or other suitable person who hears evidence at a different place and time to the actual court case. This evidence can then be used during the court case.",
        "complaint": "A statement accusing a person of breaking the law.",
        "confiscation": "Property or money taken from an offender who benefited from crime.",
        "conviction": "A pronouncement of guilt by the judge against the accused.If an accused person is convicted of an offence, this means the accused is formally declared to have committed that crime.",
        "cross-examination": "Being questioned by the lawyers of the opposing party.",
        "default sentence": "An imprisonment term served by an accused person when they fail to pay the fine imposed bythe court.",
        "disbursement": "A payment made by the lawyer on behalf of their client, and then usually claimed back from the client.",
        "discovery": "Stating the documents, under one party's control, which are relevant to the case and making them available tothe other party) so that there are no surprises when the trial starts.",
        "domicile\u00a0": "The place that a person treats as his or her permanent home, or lives in and has a substantial connection with.",
        "enter an appearance": "When the plaintiff or defendant comes to court, either in person or through a lawyer, to state or defend a claim.",
        "estate": "The money and property owned by a particular person (such as cash, real estate, financial securities, possessions and other assets), as well as their liabilities (such as debts).",
        "ex parte": "This means on behalf of or from one side only. An ex parte application would mean no other party has to be served with the application.",
        "executor": "A person named in a testator\u2019s will, appointed to carry out the terms of their will, including managing the testator\u2019s property and distributing the estate to the beneficiaries after payment of the testator\u2019s debts and other expenses.",
        "foreclosure": "The legal process in which a lender tries to recover the balance of a loan from a borrower by forcing the sale of the asset which was used as security for the loan.",
        "forensic evidence": "Scientific evidence collected from a crime scene such as DNA or fingerprints.",
        "grounds": "Reasons specified by the law that can serve as a basis for an application to the court.",
        "habitual residence": "Cases show that the courts have considered length of stay in a particular place, parents' shared intention, the child's acclimatisation, and the parents' acclimatisation to determine this term.",
        "immovable property": "Includes land, the benefits arising out of land and things attached to the earth or permanently fastened to anything attached to the earth.",
        "incapacitated": "An incapacitated person who is wholly or partially incapacitated or infirm, by reason of physical or mental disability or ill-health or old age",
        "incapacitated husband": "An incapacitated husband means a husband who, during a marriage, becomes incapacitated from earning a livelihood by any physical or mental disability or illness, becomes unable to maintain himself and continues to be unable to maintain himself.",
        "indictment": "A court document that sets out the charges that an accused person faces.",
        "inference": "An opinion or a guess based on information that the person has.",
        "inquiry": "A request for information.",
        "inter partes": "One or more parties have to be served with an application which is usually a summons or Originating Summons",
        "interlocutory": "The intermediate stage between the start and the conclusion of a legal action. A judicial decision at this stage provides a temporary or interim decision on an issue, and not on the main subject matter of the legal action.",
        "intestate succession": "The term used when a person dies without making a will.",
        "judge": "A law expert in charge of all court proceedings while ensuring legal rules are followed.",
        "judicial management": "Where an independent judicial manager is appointed by the court to manage the operations, business and property of a company which is under financial distress.",
        "judicial sales": "The forced sale of any property by an official appointed by the court to satisfy a judgment or implement another order of the court. Such sales require public notice of time, place and a description of the goods to be sold.",
        "liable": "To be legally obligated or responsible.",
        "maintenance": "The provision of support (usually financial in nature) for family members.",
        "making representations": "Representations are letters or emails where an accused person (or their lawyer) sets out clearly the circumstances of the case, other facts which are relevant to how the case may be handled, and the accused's request regarding the charge.",
        "mitigation": "A chance for an accused person to convey relevant facts (such as reasons or explanations)to the judge for leniency to be shown to the accused during sentencing.",
        "movable property": "Anything that is capable of being owned by a person or an entity that is not an immovable property.",
        "newton hearing": "An additional hearing also known as an \u201cancillary hearing\u201d, convened during the sentencingprocess when there is a dispute as to facts which may materially affect the sentence tobe imposed on an accused following his\n                conviction.",
        "next of kin": "Closest relative.",
        "notes of evidence": "A word-for-word transcript of what had been said by the different people in court.",
        "oath": "A swearing to tell the truth in court, usually according to religious beliefs, failing which the oath-taker could be prosecuted for lying.",
        "offender": "Someone who has committed an offence.",
        "plea": "The answer that an accused person gives to the court at the start of a trial when asked if guilty or not.",
        "plead guilty": "If an accused person pleads guilty, they admit to committing the offences as stated in thecharges.",
        "pleading": "A document in which a party states the facts on which they rely for their claim or defence.A Statement of Claim, Memorandum of Appearance, defence (or defence and counterclaim) and reply are types of pleadings.",
        "prayer": "A specific request for the court to do something.",
        "prosecution": "Prosecutors conduct criminal proceedings against an accused on behalf of the State.",
        "question of law": "A question about the law rather than facts of a court case.",
        "recalcitrant offender": "An individual who has been sentenced to imprisonment or corrective training multiple times.",
        "receiver": "A third party appointed by the court to take charge of the assets of a party to a lawsuit while the right to the assets is in dispute.",
        "receivership": "The process of the court appointing a receiver to take control of the assets of a party to a lawsuit pending a final decision by the court.",
        "reconyevance": "The transfer of title to the property, which was used as security for a loan, back to the borrower when the debt is fully paid.",
        "redemption": "The act of buying back the property after paying the full amount due to the lender.",
        "relief": "Any benefit the court gives a party to a lawsuit, including a monetary award, the return of property, alimony, etc.",
        "remand": "When an accused person is kept in prison or police cell pending investigations or whenthe accused cannot raise bail.",
        "remedy": "The enforcement of a legal right which a party to a lawsuit claims has been infringed upon and has caused harm.",
        "sentence": "The judge's decision on what should happen when an accused person is found guilty of breaking the law.",
        "sole executor": "The only executor named in a testator\u2019s will.",
        "statement of facts": "A statement of facts prepared by the prosecution which contains relevant facts of theoffence.",
        "stood down charge": "A charge that is temporarily put on hold, but which the prosecution may at a later stage(1) apply to take-into-consideration (TIC) for the purpose of sentencing, (2) apply toproceed with it or (3) withdraw it.",
        "substituted executor": "A person who takes over the duties of an original executor if they are unable to act as an executor of a will.",
        "summary of facts": "The document containing an accused person's version of\u00a0facts\u00a0that gave rise to their defence.",
        "summons to a witness": "A document issued by the court to a person ordering him to come to court on astipulated date and time to give evidence as a witness in a criminal case.",
        "take-into-consideration (tic) charges": "Charges considered by the judge in determining the punishment. An accused person is notpunished separately for the TIC charges, although the overall punishment imposed onthe accused may be increased as a result of the TIC charges.",
        "testator": "A person who has made a will.",
        "tort": "The law of tort is concerned with civil wrongdoings between private individuals. What is allowed between private individuals is determined by common law as determined by the courts. To some extent, this is supported by statutes.",
        "trial": "A judicial determination and examination of legal issues and facts arising between parties in a criminal, civil or family case.",
        "undertaking": "The document signed by a person who was arrested and released on bail after promising to return to court when required. This person may have to abide by certain conditions while on bail.",
        "unliquidated damages": "The amount of damages decided by the court when the parties to a contract had not specified how much the damages would be for breaching the terms of the contract.",
        "vacate a hearing": "To remove from the court calendar the hearing that was scheduled on the date set aside for it.",
        "verdict": "The decision reached at the end of a trial on whether an accused person is guilty or not.",
        "warrant": "A court document that allows the police to take certain actions, such as searching premises or arresting someone.",
        "witness": "A person who gives evidence at a trial. A witness has to formally give an oath or affirmation to tell the truth before being allowed to give evidence."
    }

    const Summarise = async () => {
        try {
            console.log("summarise in progress");
            const response = await axios.post(`http://localhost:5000/${params}`, { text: inputText.toLowerCase() }, {
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*' // Enable CORS for all origins
                }
            });
            setSummary(response.data.summary);
        } catch (error) {
            console.error('Error summarizing text:', error);
        }
    }

    return (
        <main className="flex min-h-screen flex-col items-center justify-between p-24">
            <div>
                <p className="text-bold text-xl">
                    {/* {params} */}
                </p>
            </div>
            <form className="w-full">
                <div className="w-full mb-4 border border-gray-200 rounded-lg bg-gray-50 dark:bg-gray-700 dark:border-gray-600">
                    <div className="px-4 py-2 bg-white rounded-t-lg dark:bg-gray-800">
                        <label className="sr-only">Your legal text</label>
                        <textarea
                            rows="4"
                            className="w-full px-0 text-sm text-gray-900 bg-white border-0 dark:bg-gray-800 focus:ring-0 dark:text-white dark:placeholder-gray-400"
                            placeholder="Enter legal text..."
                            value={inputText}
                            onChange={(e) => setInputText(e.target.value)}
                            required
                        />
                    </div>
                    <div className="flex items-center justify-between px-3 py-2 border-t dark:border-gray-600">
                        <button
                            onClick={Summarise}
                            type="button"
                            className="text-gray-900 bg-white border border-gray-300 focus:outline-none hover:bg-gray-100 focus:ring-4 focus:ring-gray-100 font-medium rounded-lg text-sm px-5 py-2.5 me-2 mb-2 dark:bg-gray-800 dark:text-white dark:border-gray-600 dark:hover:bg-gray-700 dark:hover:border-gray-600 dark:focus:ring-gray-700">
                            Get Summary
                        </button>
                    </div>
                </div>
            </form>
            {summary &&
                <div>
                    <section className="bg-white dark:bg-gray-900 m-3">
                        <div className="py-8 px-4 mx-auto max-w-screen-xl text-center lg:py-16">
                            <h1 className="mb-4 text-4xl font-extrabold tracking-tight leading-none text-gray-900 md:text-5xl lg:text-6xl dark:text-white">Summary of text</h1>
                            <p className="mb-8 text-lg font-normal text-gray-500 lg:text-xl sm:px-16 lg:px-48 dark:text-gray-400">{summary}</p>
                            <div className="flex flex-col space-y-4 sm:flex-row sm:justify-center sm:space-y-0">

                            </div>
                        </div>
                    </section>
                    <section className="bg-white dark:bg-gray-900 m-3">
                        <div className="py-8 px-4 mx-auto max-w-screen-xl text-center lg:py-16">
                            <h1 className="mb-4 text-4xl font-extrabold tracking-tight leading-none text-gray-900 md:text-5xl lg:text-6xl dark:text-white">Glossary</h1>
                            <p className="mb-8 text-lg font-normal text-gray-500 lg:text-xl sm:px-16 lg:px-48 dark:text-gray-400">Legal Terminology within the summary</p>
                            <div className="flex flex-col space-y-4 sm:flex-row sm:justify-center sm:space-y-0">
                                <ul>
                                    {
                                        summary.split(" ").map((word, index) => {
                                            const keyword = word.toLowerCase();
                                            if (glosarry.hasOwnProperty(keyword)) {
                                                return <li key={index}>{word}: {glosarry[keyword]}</li>
                                            } else {
                                                return null;
                                            }
                                        })
                                    }
                                </ul>
                            </div>
                        </div>
                    </section>
                </div>
            }
        </main>
    )
}