import Link from "next/link";
export default function Models() {
    const models = [
        { name: "T5", link: "https://huggingface.co/docs/transformers/en/model_doc/t5" },
        { name: "Pegasus", link: "https://huggingface.co/docs/transformers/en/model_doc/pegasus" },
        { name: "Bart", link: "https://huggingface.co/docs/transformers/en/model_doc/bart" }
    ]
    return (
        <main className="flex min-h-screen flex-col items-center justify-between p-24">
            <div className="flex">
                {
                    models.map((model) =>
                        <div key={model.name} href="#" class="block max-w-sm m-4 p-6 bg-white border border-gray-200 rounded-lg shadow hover:bg-gray-100 dark:bg-gray-800 dark:border-gray-700 dark:hover:bg-gray-700">

                            <h5 class="mb-2 text-2xl font-bold tracking-tight text-gray-900 dark:text-white">{model.name}</h5>
                            <span class="font-normal text-gray-700 dark:text-gray-400">
                                Here is the documentation on Hugging Face:&nbsp;
                            </span>
                            <Link className="underline" href={model.link}>{model.name}</Link>
                            <br /> <br />
                            <div className="items-center">
                                <Link href={`${model.name}`}>
                                    <button type="button" class="text-white bg-gray-800 hover:bg-gray-900 focus:outline-none focus:ring-4 focus:ring-gray-300 font-medium rounded-lg text-sm px-5 py-2.5 me-2 mb-2 dark:bg-gray-800 dark:hover:bg-gray-700 dark:focus:ring-gray-700 dark:border-gray-700">
                                        Try {model.name}
                                    </button>
                                </Link>
                            </div>
                        </div>
                    )
                }
            </div>
        </main>
    );
}