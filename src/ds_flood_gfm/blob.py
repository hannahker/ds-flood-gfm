import io
import ocha_stratus as stratus

def save_plot_to_blob(plt, fname):
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight", dpi=300)
        buffer.seek(0)
        container_client = stratus.get_container_client(
            "projects", "dev", write=True
        )
        blob_name = (
            f"ds-flood-gfm/plots/{fname}.png"
        )

        container_client.upload_blob(
            name=blob_name, data=buffer.getvalue(), overwrite=True
        )
        buffer.close()
        print(f"File saved on blob to {blob_name}!")
        