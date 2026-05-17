use anyhow::{Context, Result};
use jellyfin_api::{
    Client, Identity, ItemType, ItemsQuery, SortOrder, audio_stream_url, ticks_to_seconds,
    video_stream_url,
};
use url::Url;

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let server = args
        .next()
        .context("usage: poc-jellyfin-api <url> <user> <password>")?;
    let username = args.next().context("missing <user>")?;
    let password = args.next().context("missing <password>")?;

    let base = Url::parse(&server).with_context(|| format!("invalid URL: {server}"))?;
    let identity = Identity::new("Jelly", "poc-mac", "poc-device-id-fixed", "0.1");

    let mut client = Client::new(base.clone(), identity).with_accept_language("es");

    println!("[1] GET /System/Info/Public");
    let info = client.public_system_info().await?;
    println!(
        "    server={:?} version={:?} id={:?}",
        info.server_name, info.version, info.id
    );

    println!("\n[2] POST /Users/AuthenticateByName");
    let auth = client.sign_in(&username, &password).await?;
    let user = auth.user.context("auth response had no user")?;
    let token = auth.access_token.context("auth response had no token")?;
    println!(
        "    user.id={} user.name={} token={}...{}",
        user.id,
        user.name,
        &token[..4.min(token.len())],
        if token.len() > 8 {
            &token[token.len() - 4..]
        } else {
            ""
        },
    );

    println!("\n[3] GET /Users/Me");
    let me = client.current_user().await?;
    assert_eq!(me.id, user.id, "current_user should match sign_in user");
    println!("    OK — {} (id matches)", me.name);

    println!("\n[4] GET /Users/{{id}}/Views");
    let views = client.user_views(&user.id).await?;
    println!("    {} views:", views.items.len());
    for v in &views.items {
        println!(
            "      - {:?}  type={:?}  id={}",
            v.name, v.collection_type, v.id
        );
    }

    // Pick the first music library (CollectionType == "music") if any.
    let music_view = views
        .items
        .iter()
        .find(|v| v.collection_type.as_deref() == Some("music"))
        .cloned();

    if let Some(mv) = music_view {
        println!(
            "\n[5] GET /Users/{{id}}/Items  parent={}  type=MusicAlbum  limit=5",
            mv.id
        );
        let q = ItemsQuery {
            parent_id: Some(mv.id.clone()),
            include_item_types: vec![ItemType::MusicAlbum],
            sort_by: vec!["SortName".into()],
            sort_order: Some(SortOrder::Ascending),
            limit: Some(5),
            recursive: Some(true),
            fields: vec!["PrimaryImageAspectRatio".into(), "Genres".into()],
            ..Default::default()
        };
        let albums = client.items(&user.id, &q).await?;
        println!(
            "    {} of {} albums:",
            albums.items.len(),
            albums.total_record_count.unwrap_or(-1)
        );
        for a in &albums.items {
            println!(
                "      - {:?}  ({:?})  id={}",
                a.name, a.production_year, a.id
            );
        }

        if let Some(first_album) = albums.items.first() {
            println!(
                "\n[6] GET /Users/{{id}}/Items  parent={}  type=Audio  (album tracks)",
                first_album.id
            );
            let q = ItemsQuery {
                parent_id: Some(first_album.id.clone()),
                include_item_types: vec![ItemType::Audio],
                sort_by: vec!["ParentIndexNumber".into(), "IndexNumber".into()],
                limit: Some(20),
                fields: vec!["MediaSources".into(), "MediaStreams".into()],
                ..Default::default()
            };
            let tracks = client.items(&user.id, &q).await?;
            println!("    {} tracks:", tracks.items.len());
            for t in tracks.items.iter().take(5) {
                let dur = t.run_time_ticks.map(ticks_to_seconds).unwrap_or(0.0);
                let codec = t
                    .media_sources
                    .as_ref()
                    .and_then(|m| m.first())
                    .and_then(|s| s.container.clone())
                    .unwrap_or_else(|| "?".into());
                println!(
                    "      - {:>2}. {:?}  {:.0}s  ({})",
                    t.index_number.unwrap_or(0),
                    t.name,
                    dur,
                    codec
                );
            }

            if let Some(track) = tracks.items.first() {
                let stream = audio_stream_url(&base, &track.id, &token);
                println!("\n[7] audio_stream_url for first track");
                println!("    {stream}");
                // Quick HEAD to make sure the URL actually works.
                let http = reqwest::Client::new();
                let head = http.head(stream.as_str()).send().await?;
                println!(
                    "    HEAD => {}  ({} bytes per Content-Length)",
                    head.status(),
                    head.headers()
                        .get(reqwest::header::CONTENT_LENGTH)
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("?")
                );
            }
        }
    } else {
        println!("\n[5] no music library found, skipping items walk");
    }

    // Look for a video library to validate that URL too.
    if let Some(mv) = views.items.iter().find(|v| {
        matches!(
            v.collection_type.as_deref(),
            Some("movies") | Some("tvshows")
        )
    }) {
        let q = ItemsQuery {
            parent_id: Some(mv.id.clone()),
            include_item_types: if mv.collection_type.as_deref() == Some("movies") {
                vec![ItemType::Movie]
            } else {
                vec![ItemType::Series]
            },
            limit: Some(1),
            recursive: Some(true),
            ..Default::default()
        };
        let one = client.items(&user.id, &q).await?;
        if let Some(item) = one.items.first() {
            let stream = video_stream_url(&base, &item.id, &token);
            println!("\n[8] video_stream_url sample ({:?})", item.name);
            println!("    {stream}");
        }
    }

    println!("\n[done]");
    Ok(())
}
