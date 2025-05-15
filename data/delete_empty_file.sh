#!/bin/bash

# chord フォルダ内のすべての .json ファイルを処理
for json_file in chord/*.json; do
  # ファイルの中身を読み取り
  content=$(<"$json_file")
  
  # 中身が "{}" だけの場合
  if [[ "$content" == "{}" ]]; then
    # ベースファイル名を取得（拡張子なし）
    base_name=$(basename "$json_file" .json)
    
    # 該当の mp3 ファイル
    mp3_file="audio/$base_name.mp3"
    
    # ファイルを削除
    echo "Deleting: $json_file"
    rm "$json_file"
    
    if [[ -f "$mp3_file" ]]; then
      echo "Deleting: $mp3_file"
      rm "$mp3_file"
    else
      echo "MP3 file not found: $mp3_file"
    fi
  fi
done
